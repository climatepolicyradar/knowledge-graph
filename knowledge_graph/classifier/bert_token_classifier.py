import logging
import os
import random
import re
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing_extensions import Self

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ProbabilityCapableClassifier,
    VariantEnabledClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span

logging.basicConfig(handlers=[RichHandler()])
logger = logging.getLogger(__name__)

# BIO label scheme
O_LABEL = 0
B_LABEL = 1
I_LABEL = 2
IGNORE_LABEL = -100
NUM_LABELS = 3

ID2LABEL = {O_LABEL: "O", B_LABEL: "B-CONCEPT", I_LABEL: "I-CONCEPT"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def _align_labels_with_tokens(
    offset_mapping: list[tuple[int, int]],
    spans: list[Span],
    concept_id: WikibaseID | None,
) -> list[int]:
    """
    Align character-level span annotations to subword token positions using BIO tagging.

    For each token, determines whether it falls inside a gold span for the given concept,
    and assigns a BIO label accordingly.

    :param offset_mapping: List of (start_char, end_char) tuples from the tokenizer,
        one per token. Special tokens have (0, 0).
    :param spans: Gold span annotations for this passage.
    :param concept_id: The concept ID to filter spans by.
    :returns: List of integer labels, one per token.
    """
    # Filter to spans for this concept
    concept_spans = [s for s in spans if s.concept_id == concept_id]

    # Sort spans by start_index so we can track B vs I correctly
    concept_spans.sort(key=lambda s: s.start_index)

    labels = []
    prev_span_idx: int | None = None  # which span the previous real token was in

    for tok_start, tok_end in offset_mapping:
        # Special tokens (CLS, SEP, PAD) have (0, 0) offset
        if tok_start == 0 and tok_end == 0:
            labels.append(IGNORE_LABEL)
            prev_span_idx = None
            continue

        # Find which span this token overlaps with (if any)
        current_span_idx = None
        for i, span in enumerate(concept_spans):
            if tok_start < span.end_index and tok_end > span.start_index:
                current_span_idx = i
                break

        if current_span_idx is not None:
            if current_span_idx == prev_span_idx:
                labels.append(I_LABEL)
            else:
                labels.append(B_LABEL)
            prev_span_idx = current_span_idx
        else:
            labels.append(O_LABEL)
            prev_span_idx = None

    return labels


def _reconstruct_spans_from_predictions(
    token_labels: list[int],
    token_probs: list[float],
    offset_mapping: list[tuple[int, int]],
    text: str,
    concept_id: WikibaseID | None,
    labeller: str,
    min_span_chars: int = 2,
) -> list[Span]:
    """
    Convert BIO token predictions back to character-level Span objects.

    Merges consecutive B/I tokens into contiguous spans and averages their
    probabilities.

    :param token_labels: Predicted BIO label per token.
    :param token_probs: Probability of the predicted label per token.
    :param offset_mapping: (start_char, end_char) per token from the tokenizer.
    :param text: The original text.
    :param concept_id: Concept ID to assign to spans.
    :param labeller: Labeller string for the spans.
    :param min_span_chars: Minimum span length in characters; shorter spans are dropped.
    :returns: List of reconstructed Span objects.
    """
    spans: list[Span] = []
    now = datetime.now()

    current_start: int | None = None
    current_end: int | None = None
    current_probs: list[float] = []

    def _finalise_span():
        nonlocal current_start, current_end, current_probs
        if current_start is not None and current_end is not None:
            if current_end - current_start >= min_span_chars:
                spans.append(
                    Span(
                        text=text,
                        start_index=current_start,
                        end_index=current_end,
                        concept_id=concept_id,
                        prediction_probability=float(np.mean(current_probs)),
                        labellers=[labeller],
                        timestamps=[now],
                    )
                )
        current_start = None
        current_end = None
        current_probs = []

    for label, prob, (tok_start, tok_end) in zip(
        token_labels, token_probs, offset_mapping
    ):
        # Skip special tokens
        if tok_start == 0 and tok_end == 0:
            continue

        if label == B_LABEL:
            # Finalise any open span before starting a new one
            _finalise_span()
            current_start = tok_start
            current_end = tok_end
            current_probs = [prob]
        elif label == I_LABEL and current_start is not None:
            # Extend current span
            current_end = tok_end
            current_probs.append(prob)
        else:
            # O label or orphaned I label: finalise any open span
            _finalise_span()

    # Finalise any span still open at end of sequence
    _finalise_span()

    return spans


def _count_bio_labels(label_sequences: list[list[int]]) -> dict[int, int]:
    """Count occurrences of each BIO label across all sequences."""
    counts: dict[int, int] = {O_LABEL: 0, B_LABEL: 0, I_LABEL: 0}
    for seq in label_sequences:
        for label in seq:
            if label in counts:
                counts[label] += 1
    return counts


def _compute_token_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute token-level metrics from token predictions.

    Note: precision/recall/F1 here are token-level, not span-level. A 5-token
    entity counts as 5 positives. This is used for early stopping during
    training; the production quality signal is the span-level Jaccard metric
    computed by evaluate_classifier.
    """
    predictions = np.argmax(eval_pred.predictions, axis=2)  # [batch, seq_len]
    labels = eval_pred.label_ids  # [batch, seq_len]

    # Flatten, ignoring positions with IGNORE_LABEL
    mask = labels != IGNORE_LABEL
    flat_preds = predictions[mask]
    flat_labels = labels[mask]

    accuracy = accuracy_score(flat_labels, flat_preds)

    # Token-level positives: any non-O token
    pred_positive = flat_preds != O_LABEL
    gold_positive = flat_labels != O_LABEL

    tp = int(np.sum(pred_positive & gold_positive))
    fp = int(np.sum(pred_positive & ~gold_positive))
    fn = int(np.sum(~pred_positive & gold_positive))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


class WeightedTokenTrainer(Trainer):
    """Trainer that applies class weights to token-level cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute weighted token-level cross-entropy loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=IGNORE_LABEL
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

        # Reshape: [batch, seq_len, num_labels] -> [batch*seq_len, num_labels]
        loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class BertTokenClassifier(
    Classifier,
    GPUBoundClassifier,
    VariantEnabledClassifier,
    ProbabilityCapableClassifier,
):
    """
    Token-level BERT classifier.

    Performs BIO (beginning-inside-outside) token classification to identify precisely
    which tokens mention the concept.

    https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
    """

    def __init__(
        self,
        concept: Concept,
        model_name: str = "answerdotai/ModernBERT-base",
        download_pretrained_model_on_init: bool = True,
        unfreeze_layers: int = 2,
    ):
        super().__init__(concept)
        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers

        self._use_dropout_during_inference = False
        self._variant_seed = False
        self._variant_dropout_rate = False
        self._random_id_component: str = ""

        self.device = self._resolve_device()

        if download_pretrained_model_on_init:
            self.download_model_and_tokenizer()

    def _resolve_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def move_model_to_device(self, device: torch.device | None = None) -> None:
        """Move a model to a chosen device"""
        device = device or self._resolve_device()
        if hasattr(self.model, "to"):
            self.model.to(device)  # type: ignore[arg-type]
        self.device = device

    def download_model_and_tokenizer(self) -> None:
        """Download pretrained base model and tokenizer"""
        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        if not self.tokenizer.is_fast:
            raise ValueError(
                f"Tokenizer for {self.model_name} is not a fast tokenizer. "
                "A fast tokenizer is required for offset_mapping support."
            )

        self.model.to(self.device)  # type: ignore[arg-type]

    @property
    def id(self) -> ClassifierID:
        """Classifier ID"""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.model_name,
            self.prediction_threshold,
            self.unfreeze_layers,
            self._random_id_component,
        )

    @contextmanager
    def _dropout_enabled(self):
        was_training = self.model.training

        dropout_layers = [
            m for m in self.model.modules() if isinstance(m, torch.nn.Dropout)
        ]

        if not hasattr(self, "_dropout_logged"):
            if not dropout_layers:
                logger.warning(
                    "No dropout layers found in model %s. "
                    "Ensemble variants may produce identical predictions.",
                    self.model_name,
                )
            else:
                original_rates = {layer.p for layer in dropout_layers}
                logger.info(
                    "Found %d dropout layers with original rates: %s",
                    len(dropout_layers),
                    original_rates,
                )
            self._dropout_logged = True

        original_dropout_rates = [layer.p for layer in dropout_layers]

        if self._variant_dropout_rate is not None and dropout_layers:
            for layer in dropout_layers:
                layer.p = self._variant_dropout_rate

        if self._variant_seed is not None:
            torch.manual_seed(self._variant_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._variant_seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self._variant_seed)

        self.model.train()
        try:
            yield
        finally:
            for layer, original_rate in zip(dropout_layers, original_dropout_rates):
                layer.p = original_rate
            self.model.train(was_training)

    def get_variant(
        self, random_seed: int | None = None, dropout_rate: float = 0.1
    ) -> Self:
        """Get a variant of the classifier for Monte Carlo dropout estimation."""
        variant = self.__class__(
            concept=self.concept,
            model_name=self.model_name,
            download_pretrained_model_on_init=True,
            unfreeze_layers=self.unfreeze_layers,
        )
        variant.model.load_state_dict(self.model.state_dict())
        variant.tokenizer = self.tokenizer
        variant.device = self.device

        variant._variant_seed = random_seed
        variant._variant_dropout_rate = dropout_rate
        variant._use_dropout_during_inference = True
        variant.is_fitted = self.is_fitted

        return variant

    def _prepare_dataset(self, labelled_passages: list[LabelledPassage]) -> Dataset:
        texts = [passage.text for passage in labelled_passages]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors=None,
            return_offsets_mapping=True,
        )

        all_labels = []
        for i, passage in enumerate(labelled_passages):
            offset_mapping = tokenized["offset_mapping"][i]  # type: ignore[index]
            labels = _align_labels_with_tokens(
                offset_mapping=offset_mapping,
                spans=passage.spans,
                concept_id=self.concept.wikibase_id,
            )
            all_labels.append(labels)

        # Remove offset_mapping before creating the dataset (model doesn't need it)
        tokenized.pop("offset_mapping")

        return Dataset.from_dict({**tokenized, "labels": all_labels})

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        return self._predict_batch([text], threshold=threshold)[0]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        if self._use_dropout_during_inference:
            with self._dropout_enabled():
                with torch.no_grad():
                    batch_labels, batch_probs, batch_offsets = self._forward(texts)
        else:
            self.model.eval()
            with torch.no_grad():
                batch_labels, batch_probs, batch_offsets = self._forward(texts)

        effective_threshold = (
            threshold if threshold is not None else self.prediction_threshold
        )

        labeller = str(self)
        results = []
        for text, token_labels, token_probs, offsets in zip(
            texts, batch_labels, batch_probs, batch_offsets
        ):
            spans = _reconstruct_spans_from_predictions(
                token_labels=token_labels,
                token_probs=token_probs,
                offset_mapping=offsets,
                text=text,
                concept_id=self.concept.wikibase_id,
                labeller=labeller,
            )

            if effective_threshold is not None:
                spans = [
                    s
                    for s in spans
                    if s.prediction_probability is not None
                    and s.prediction_probability >= effective_threshold
                ]

            results.append(spans)

        return results

    def _forward(
        self, texts: Sequence[str]
    ) -> tuple[list[list[int]], list[list[float]], list[list[tuple[int, int]]]]:
        """
        Run a batched forward pass for token classification.

        :returns: Tuple of (batch_labels, batch_probs, batch_offsets) where each
            is a list-per-text of per-token values.
        """
        device = self.device
        tokenized = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # offset_mapping is not a model input
        offset_mapping = tokenized.pop("offset_mapping").tolist()

        inputs = {k: v.to(device) for k, v in tokenized.items()}
        logits = self.model(**inputs).logits  # [batch, seq_len, num_labels]

        probs = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)  # [batch, seq_len]

        # Get the probability of the predicted class for each token
        batch_size, seq_len, _ = probs.shape
        predicted_probs = probs[
            torch.arange(batch_size).unsqueeze(1),
            torch.arange(seq_len).unsqueeze(0),
            predicted_classes,
        ]

        batch_labels = predicted_classes.tolist()
        batch_probs = predicted_probs.tolist()

        return batch_labels, batch_probs, offset_mapping

    def fit(
        self,
        labelled_passages: list[LabelledPassage],
        validation_size: float = 0.2,
        enable_wandb: bool = False,
        **kwargs,
    ) -> "BertTokenClassifier":
        """
        Fine tune the base model on some labelled passages.

        :param labelled_passages: Training data.
        :param validation_size: The proportion of labelled passages to use for validation.
        :param bool enable_wandb: Whether to enable W&B logging for training metrics and model checkpoints.
        :return BertTokenClassifier: The trained classifier.
        """
        if len(labelled_passages) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for "
                f"{self.concept.wikibase_id}. At least 10 are required."
            )

        # Passage-level labels for stratified splitting
        labels = [
            1
            if any(
                (span.concept_id == self.concept.wikibase_id)
                and (span.prediction_probability != 0)
                for span in p.spans
            )
            else 0
            for p in labelled_passages
        ]

        train_passages, val_passages, _, _ = train_test_split(
            labelled_passages,
            labels,
            test_size=validation_size,
            random_state=42,
            stratify=labels,
        )

        logger.info("Preparing datasets...")
        train_dataset = self._prepare_dataset(train_passages)
        validation_dataset = self._prepare_dataset(val_passages)

        # Dataset statistics
        train_label_counts = _count_bio_labels(train_dataset["labels"])
        val_label_counts = _count_bio_labels(validation_dataset["labels"])

        stats_df = pd.DataFrame(
            {
                "Split": ["Training", "Validation"],
                "Passages": [len(train_dataset), len(validation_dataset)],
                "B tokens": [train_label_counts[B_LABEL], val_label_counts[B_LABEL]],
                "I tokens": [train_label_counts[I_LABEL], val_label_counts[I_LABEL]],
                "O tokens": [train_label_counts[O_LABEL], val_label_counts[O_LABEL]],
            }
        )

        logger.info(stats_df.to_markdown(index=False, tablefmt="rounded_grid"))

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
                "Freezing base model. The token classification head will be trained."
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

        # Compute token-level class weights
        all_train_labels = [
            label
            for seq in train_dataset["labels"]
            for label in seq
            if label != IGNORE_LABEL
        ]
        unique_labels = sorted(set(all_train_labels))
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(unique_labels),
            y=np.array(all_train_labels),
        )

        # Build full weight tensor for all NUM_LABELS classes
        weight_tensor = torch.ones(NUM_LABELS, dtype=torch.float32)
        for label, weight in zip(unique_labels, class_weights):
            weight_tensor[label] = weight
        weight_tensor = weight_tensor.to(self.device)
        logger.info("Class weights (O, B, I): %s", weight_tensor.cpu().numpy())

        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=os.path.join(temp_dir, "results"),
                num_train_epochs=10,
                per_device_train_batch_size=min(32, max(8, len(train_dataset) // 10)),
                per_device_eval_batch_size=32,
                max_grad_norm=1.0,
                learning_rate=2e-4,
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
                run_name=(f"{self.concept.id}_{self.name}" if enable_wandb else None),
                log_level="info" if enable_wandb else "warning",
            )

            trainer = WeightedTokenTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=_compute_token_metrics,
                class_weights=weight_tensor,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=2, early_stopping_threshold=0
                    )
                ],
            )

            logger.info("Starting training...")
            trainer.train()

            if enable_wandb:
                import wandb as _wandb

                if _wandb.run is not None:
                    _wandb.config.update(
                        {"unfreeze_layers": self.unfreeze_layers}, allow_val_change=True
                    )

            logger.info("Evaluating on validation set...")
            eval_results = trainer.evaluate(eval_dataset=validation_dataset)  # type: ignore[arg-type]

            results_df = pd.DataFrame(
                {
                    "Metric": ["F1 Score", "Accuracy", "Precision", "Recall"],
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
            logger.info("Final F1 score: %.4f", final_f1)

        logger.info("Training complete for concept %s!", self.concept.id)

        self.is_fitted = True
        self._random_id_component = str(random.getrandbits(128))

        return self
