"""Benchmark the effect of max_length on BERT classifier training and inference speed."""

import os
import random
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import typer
import wandb
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments

import knowledge_graph.classifier.bert_based as bert_based
from knowledge_graph.concept import Concept
from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.utils import deserialise_pydantic_list_from_jsonl

logger = bert_based.logger

console = Console()
app = typer.Typer()

SAMPLED_PASSAGES_DIR = processed_data_dir / "sampled_passages"
MODEL_NAME = "answerdotai/ModernBERT-base"

# Token length distribution from production corpus (Snowflake, 2026-04-21)
TARGET_DISTRIBUTION: dict[str, float] = {
    "1-256": 0.9464,
    "257-512": 0.0350,
    "513-1024": 0.0134,
    "1025-2048": 0.0044,
    "2049-4096": 0.0006,
    "4097-8192": 0.0001,
}

BUCKET_BOUNDS: list[tuple[str, int, int]] = [
    ("1-256", 1, 256),
    ("257-512", 257, 512),
    ("513-1024", 513, 1024),
    ("1025-2048", 1025, 2048),
    ("2049-4096", 2049, 4096),
    ("4097-8192", 4097, 8192),
]

# Minimum number of passages in each bucket as there are too few passages in some buckets.
MINIMUM_BUCKET_COUNTS: dict[str, int] = {
    "1025-2048": 50,
    "2049-4096": 50,
    "4097-8192": 50,
}


@dataclass
class BenchmarkConfig:
    """Configuration for the max length benchmark."""

    name: str
    max_length: int
    use_dynamic_padding: bool


BENCHMARK_CONFIGS = [
    BenchmarkConfig("8192_dynamic", max_length=8192, use_dynamic_padding=True),
    BenchmarkConfig("4096_dynamic", max_length=4096, use_dynamic_padding=True),
    BenchmarkConfig("2048_dynamic", max_length=2048, use_dynamic_padding=True),
    BenchmarkConfig("1024_dynamic", max_length=1024, use_dynamic_padding=True),
    BenchmarkConfig("512_dynamic", max_length=512, use_dynamic_padding=True),
    BenchmarkConfig("512_static", max_length=512, use_dynamic_padding=False),
]


def _get_bucket(n_tokens: int) -> str:
    for name, lo, hi in BUCKET_BOUNDS:
        if lo <= n_tokens <= hi:
            return name
    return BUCKET_BOUNDS[-1][0]


def _load_all_passages(passages_dir: Path) -> list[LabelledPassage]:
    passages = []
    for jsonl_file in sorted(passages_dir.glob("*.jsonl")):
        passages.extend(
            deserialise_pydantic_list_from_jsonl(
                jsonl_file.read_text(), LabelledPassage
            )
        )
    return passages


def _generate_passage_for_bucket(
    bucket_name: str,
    all_passages: list[LabelledPassage],
    tokenizer: PreTrainedTokenizer,
    rng: random.Random,
) -> LabelledPassage:
    """Concatenate random passages to a random target length within the bucket's token range, then decode back to text when a bucket has no real data."""
    _, lo, hi = next(b for b in BUCKET_BOUNDS if b[0] == bucket_name)
    target_tokens = rng.randint(lo, hi)

    token_ids: list[int] = []
    while len(token_ids) < target_tokens - 2:  # -2 reserves space for special tokens
        source = rng.choice(all_passages)
        token_ids.extend(tokenizer.encode(source.text, add_special_tokens=False))

    text = tokenizer.decode(token_ids[: target_tokens - 2])
    return LabelledPassage(text=text, spans=[])


def build_benchmark_dataset(
    n_samples: int = 2000,
    random_seed: int = 42,
) -> tuple[list[LabelledPassage], dict[str, int]]:
    """
    Load all sampled passages and resample to match the production token-length distribution.

    Buckets with fewer passages than needed are sampled with replacement.
    Empty buckets are filled by concatenating passages from other buckets.
    """
    rng = random.Random(random_seed)

    console.log(f"Loading passages from {SAMPLED_PASSAGES_DIR}...")
    all_passages = _load_all_passages(SAMPLED_PASSAGES_DIR)
    console.log(f"Loaded {len(all_passages)} passages")

    console.log("Tokenizing to get true token lengths (no truncation)...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_counts = [
        len(tokenizer.encode(p.text, add_special_tokens=True)) for p in all_passages
    ]

    bucketed: dict[str, list[LabelledPassage]] = defaultdict(list)
    for passage, n_tokens in zip(all_passages, token_counts):
        bucketed[_get_bucket(n_tokens)].append(passage)

    table = Table(title="Source passage distribution", show_lines=True)
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    table.add_column("Actual %", justify="right")
    table.add_column("Target %", justify="right")
    for name, _, _ in BUCKET_BOUNDS:
        count = len(bucketed.get(name, []))
        table.add_row(
            name,
            str(count),
            f"{count / len(all_passages) * 100:.2f}%",
            f"{TARGET_DISTRIBUTION[name] * 100:.2f}%",
        )
    console.print(table)

    result: list[LabelledPassage] = []
    for bucket_name, target_prop in TARGET_DISTRIBUTION.items():
        target_count = max(
            round(n_samples * target_prop), MINIMUM_BUCKET_COUNTS.get(bucket_name, 0)
        )

        available = bucketed.get(bucket_name, [])

        if target_count == 0:
            continue

        if not available:
            console.log(
                f"  {bucket_name}: generating {target_count} synthetic passages"
            )
            generated = [
                _generate_passage_for_bucket(bucket_name, all_passages, tokenizer, rng)
                for _ in range(target_count)
            ]
            result.extend(generated)
            continue

        with_replacement = target_count > len(available)
        sampled = (
            rng.choices(available, k=target_count)
            if with_replacement
            else rng.sample(available, k=target_count)
        )
        result.extend(sampled)

        label = "with replacement" if with_replacement else "without replacement"
        console.log(f"  {bucket_name}: sampled {target_count} ({label})")

    rng.shuffle(result)
    console.log(f"\nBenchmark dataset: [bold]{len(result)} passages[/]")
    bucket_counts = {name: len(bucketed.get(name, [])) for name, _, _ in BUCKET_BOUNDS}
    return result, bucket_counts


class PassageLengthBenchmarkClassifier(bert_based.BertBasedClassifier):
    """A BERT-based classifier that allows for changing the max length of the passages during training and inference."""

    def __init__(self, *args, inference_max_length: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_max_length = inference_max_length

    def _prepare_dataset(
        self,
        labelled_passages: list[LabelledPassage],
        padding: bool = False,
        max_length: int = 512,
    ) -> Dataset:
        """
        Prepare a dataset from labelled passages for training with no maximum length.  Passages are not truncated.

        Args:
            labelled_passages: List of labelled passages to prepare dataset from
            padding: Whether to pad the dataset to the max length of the longest passage in the batch.
            max_length: The maximum length after which passages are truncated.

        Returns:
            Dataset: A HuggingFace dataset ready for training
        """
        texts = [passage.text for passage in labelled_passages]
        labels = [i % 2 for i in range(len(labelled_passages))]

        tokenized_inputs = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        return Dataset.from_dict({**tokenized_inputs, "labels": labels})

    # override the fit method to allow for changing the max length
    def fit(
        self,
        labelled_passages: list[LabelledPassage],
        validation_size: float = 0.2,
        enable_wandb: bool = False,
        *,
        max_length: int = 512,
        use_dynamic_padding: bool = True,
        **kwargs,
    ) -> "PassageLengthBenchmarkClassifier":
        """
        Train a BERT-based classifier on a list of labelled passages with a given max length.

        If use_dynamic_padding is True, the dataset is padded dynamically to the max length of the longest passage in the batch.
        If use_dynamic_padding is False, the dataset is padded to the max length.
        """
        if len(labelled_passages) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for "
                f"{self.concept.wikibase_id}. At least 10 are required."
            )
        # generate random labels for the dataset
        labels = [i % 2 for i in range(len(labelled_passages))]

        # Split passages into training and validation sets. Stratify to maintain the
        # distribution of positive and negative passages.
        train_passages, val_passages, _, _ = train_test_split(
            labelled_passages,
            labels,
            test_size=validation_size,
            random_state=42,
            stratify=labels,
        )

        logger.info("Preparing datasets...")
        # if use_dynamic_padding is False, we pad the dataset to the max length when preparing it.  If use_dynamic_padding is True, we pad dynamically to the max length of the longest passage in the batch when training.
        train_dataset = self._prepare_dataset(
            train_passages, padding=not use_dynamic_padding, max_length=max_length
        )
        validation_dataset = self._prepare_dataset(
            val_passages, padding=not use_dynamic_padding, max_length=max_length
        )

        # Create a summary DataFrame for dataset statistics
        stats_df = pd.DataFrame(
            {
                "Split": ["Training", "Validation"],
                "Rows": [
                    len(train_dataset),
                    len(validation_dataset),
                ],
                "Positive %": [
                    (sum(train_dataset["labels"]) / len(train_dataset)) * 100,
                    (sum(validation_dataset["labels"]) / len(validation_dataset)) * 100,
                ],
            }
        )

        logger.info(
            pd.DataFrame(stats_df)
            .round(2)
            .to_markdown(index=False, tablefmt="rounded_grid")
        )

        # Determine which transformer layers to unfreeze (if any)
        unfrozen_layer_indices: set[int] = set()
        if self.unfreeze_layers > 0:
            layer_indices = set()
            for name, _ in self.model.named_parameters():
                if match := re.search(r"layers?\.(\d+)\.", name):
                    layer_indices.add(int(match.group(1)))
                else:
                    logger.warning(f"No layers found in the model with name {name}")
            if layer_indices:
                max_layer = max(layer_indices)
                unfrozen_layer_indices = {
                    i
                    for i in layer_indices
                    if i >= max_layer - self.unfreeze_layers + 1
                }

        if unfrozen_layer_indices:
            logger.info(
                "Unfreezing transformer layers %s (plus classification head).",
                sorted(unfrozen_layer_indices),
            )
        else:
            logger.info(
                "Freezing base model. The model's prediction head and classifier will be trained."
            )

        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
            elif unfrozen_layer_indices:
                match = re.search(r"layers?\.(\d+)\.", name)
                if match and int(match.group(1)) in unfrozen_layer_indices:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Count the number of trainable parameters and display the percentage of the model
        # that is trainable after freezing the backbone
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info("  Trainable parameters: %s", f"{trainable_params:,}")
        logger.info("  Frozen parameters: %s", f"{total_params - trainable_params:,}")
        logger.info(
            "  Training %s%% of model",
            f"{trainable_params / total_params * 100:.2f}",
        )

        # Compute class weights to handle class imbalance
        train_labels = np.array(train_dataset["labels"])
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            self.device
        )
        logger.info("Class weights: %s", class_weights_tensor.cpu().numpy())

        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=os.path.join(temp_dir, "results"),
                # high number of train epochs as we enable early stopping below
                num_train_epochs=10,
                # batch size scales with dataset size, to avoid batches or epochs that
                # have too few batches which leads to unstable training
                per_device_train_batch_size=min(64, max(16, len(train_dataset) // 10)),
                per_device_eval_batch_size=64,
                # gradient clipping for more stable updates
                max_grad_norm=1.0,
                learning_rate=2e-4 if self.unfreeze_layers > 0 else 5e-4,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                optim="adamw_torch_fused",
                fp16=False,
                dataloader_pin_memory=False,
                logging_dir=os.path.join(temp_dir, "logs"),
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                dataloader_num_workers=2,
                report_to=["wandb"] if enable_wandb else [],
                disable_tqdm=True,
                # W&B-specific settings when enabled
                run_name=f"{self.concept.id}_{self.name}" if enable_wandb else None,
                log_level="info" if enable_wandb else "warning",
            )

            trainer = bert_based.WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=bert_based.compute_metrics,
                class_weights=class_weights_tensor,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=2, early_stopping_threshold=0
                    )
                ],
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
                if use_dynamic_padding
                else None,  # optional dynamic padding
            )

            logger.info("🚀 Starting training...")
            trainer.train()

            logger.info("🎯 Evaluating on validation set...")
            eval_results = trainer.evaluate(eval_dataset=validation_dataset)  # type: ignore

            results_df = pd.DataFrame(
                {
                    "Metric": [
                        "F1 Score",
                        "Accuracy",
                        "Precision",
                        "Recall",
                    ],
                    "Value": [
                        eval_results.get("eval_f1", 0),
                        eval_results.get("eval_accuracy", 0),
                        eval_results.get("eval_precision", 0),
                        eval_results.get("eval_recall", 0),
                    ],
                }
            )
            logger.info("Final Validation Results")
            logger.info(
                results_df.round(4).to_markdown(index=False, tablefmt="rounded_grid")
            )

            final_f1 = eval_results.get("eval_f1", 0)
            logger.info("📊 Final F1 score: %.4f", final_f1)

        logger.info("✅ Training complete for concept %s!", self.concept.id)

        self.is_fitted = True

        # Generate a random component for the classifier ID
        # This ensures the ID is unique to this trained instance but remains stable
        # across pickle serialization/deserialization
        self._random_id_component = str(random.getrandbits(128))

        return self

    # override the _forward method to allow for changing the max length
    def _forward(self, texts: Sequence[str]) -> tuple[list[float], list[int]]:
        """
        Run a batched forward pass and return per-text scores and predicted classes.

        :param texts: Texts to classify.
        :param max_length: The maximum length after which passages are truncated.
        :returns: Tuple of `(scores, predicted_classes)` where `scores[i]` is the
            probability of the predicted class for text `i`
        """
        device = self.device
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.inference_max_length,
            return_tensors="pt",
        ).to(device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        scores = probs[torch.arange(len(probs)), predicted_classes]
        return scores.tolist(), predicted_classes.tolist()


def run_training_benchmark(
    dataset: list[LabelledPassage],
    max_length: int = 64,
    use_dynamic_padding: bool = True,
    enable_wandb: bool = False,
    n_samples: int = 2000,
    bucket_counts: dict[str, int] | None = None,
) -> None:
    """Run a benchmark for a given max length."""

    dummy_concept = Concept(
        preferred_label="benchmark", wikibase_id=WikibaseID("Q123456")
    )

    if enable_wandb:
        wandb.init(
            project="max-length-experiment",
            config={
                "max_passage_tokens_before_truncation": max_length,
                "use_dynamic_padding": use_dynamic_padding,
                "bucket_counts": bucket_counts,
                "n_samples": n_samples,
            },
        )
    classifier = PassageLengthBenchmarkClassifier(dummy_concept)
    classifier.fit(
        dataset,
        max_length=max_length,
        use_dynamic_padding=use_dynamic_padding,
        enable_wandb=enable_wandb,
    )


def run_inference_benchmark(
    dataset: list[LabelledPassage],
    max_length: int = 64,
    enable_wandb: bool = False,
    bucket_counts: dict[str, int] | None = None,
    n_samples: int = 2000,
) -> None:
    """Run a benchmark for a given max length."""
    dummy_concept = Concept(
        preferred_label="benchmark", wikibase_id=WikibaseID("Q123456")
    )
    raw_text_passages = [passage.text for passage in dataset]

    classifier = PassageLengthBenchmarkClassifier(
        dummy_concept, inference_max_length=max_length
    )
    classifier.predict(raw_text_passages[:10], batch_size=10)
    if enable_wandb:
        wandb.init(
            project="max-length-experiment",
            config={
                "max_passage_tokens_before_truncation": max_length,
                "step": "inference",
                "bucket_counts": bucket_counts,
                "n_samples": n_samples,
            },
        )
    start_time = time.perf_counter()
    classifier.predict(raw_text_passages, batch_size=16)
    elapsed = time.perf_counter() - start_time

    if enable_wandb:
        wandb.log(
            {
                "inference_time": elapsed,
                "inference_samples_per_second": len(raw_text_passages) / elapsed,
            }
        )
        wandb.finish()


@app.command()
def main(
    n_samples: int = 2000,
    enable_wandb: bool = False,
):
    console.log(f"Building benchmark dataset with {n_samples} samples...")
    dataset, bucket_counts = build_benchmark_dataset(n_samples=n_samples)

    total = len(BENCHMARK_CONFIGS)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Running benchmarks...", total=total * 2
        )  # *2 for train + inference
        for config in BENCHMARK_CONFIGS:
            progress.update(task, description=f"Training {config.name}...")
            run_training_benchmark(
                dataset,
                config.max_length,
                config.use_dynamic_padding,
                enable_wandb=enable_wandb,
                n_samples=n_samples,
                bucket_counts=bucket_counts,
            )

            progress.advance(task)

            progress.update(task, description=f"Inference {config.name}...")
            run_inference_benchmark(
                dataset,
                config.max_length,
                enable_wandb=enable_wandb,
                bucket_counts=bucket_counts,
                n_samples=n_samples,
            )
            progress.advance(task)


if __name__ == "__main__":
    app()
