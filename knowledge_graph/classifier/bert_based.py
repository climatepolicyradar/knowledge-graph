import logging
import os
import random
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
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
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span

logging.basicConfig(handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for model evaluation"""
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the cross-entropy loss function."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute cross-entropy loss weighted by class weights provided to the trainer."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Apply class weights to the loss
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class BertBasedClassifier(
    Classifier,
    GPUBoundClassifier,
    VariantEnabledClassifier,
    ProbabilityCapableClassifier,
):
    """
    Classifier that uses a fine-tuned transformer model to identify concepts in text.

    This classifier uses a pre-trained transformer model, fine-tuned on labelled
    passages to identify instances of a concept. The model is trained to produce binary
    predictions, ie whether a given text contains an instance of the concept but not
    where the concept is specifically mentioned in the given text.

    This approach is particularly useful for identifying complex or nuanced instances
    of concepts that may not be easily captured by keyword matching or rule-based
    approaches.
    """

    def __init__(
        self,
        concept: Concept,
        model_name: str = "answerdotai/ModernBERT-base",
        download_pretrained_model_on_init: bool = True,
        unfreeze_layers: int = 0,
    ):
        """
        Initialise a BERT classifier.

        :param concept: concept the classifier is trained to detect mentions of in text
        :param model_name: model name from Huggingface, defaults to "answerdotai/ModernBERT-base"
        :param download_pretrained_model_on_init: whether to download the pretrained model and tokenizer on init, defaults to True.
            Disable this if planning to overwrite the model and tokenizer elsewhere.
        :param unfreeze_layers: number of final encoder layers to fine-tune in addition
            to the classification head. 0 keeps the full backbone frozen (head-only
            training).
        """

        super().__init__(concept)
        self.model_name = model_name
        self.unfreeze_layers = unfreeze_layers

        # Private properties for creating and running inference on classifier variants
        self._use_dropout_during_inference = False
        self._variant_seed = False
        self._variant_dropout_rate = False

        # Random component for nondeterministic ID, generated once when classifier is fitted
        self._random_id_component: str = ""

        self.device = self._resolve_device()

        if download_pretrained_model_on_init:
            self.download_model_and_tokenizer()

    def _resolve_device(self) -> torch.device:
        """
        Resolve the best available device for this machine.

        :returns: The best available torch device (MPS, CUDA, or CPU).
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def move_model_to_device(self, device: torch.device | None = None) -> None:
        """
        Move the model to the given device and update device references.

        If no device is provided, the best available device is resolved
        automatically.  This is useful after deserialisation because pickle
        does not preserve MPS/CUDA tensor placement.

        :param device: Target device. Resolved automatically when ``None``.
        """
        device = device or self._resolve_device()
        if hasattr(self.model, "to"):
            self.model.to(device)  # type: ignore[attr-defined]
        self.device = device

    def download_model_and_tokenizer(self) -> None:
        """
        Download the model and tokenizer from the Huggingface hub.

        Uses the class's `model_name` property.
        """

        if "ModernBERT" in self.model_name:
            extra_clf_kwargs = {
                # `reference_compile=False` disables ModernBERT's torch.compile path, which
                # would otherwise require a C compiler at runtime (absent from our slim image)
                "reference_compile": False,
            }
        else:
            extra_clf_kwargs = {}

        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **extra_clf_kwargs,
            )
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        self.model.to(self.device)  # type: ignore

    @property
    def id(self) -> ClassifierID:
        """
        Return a human-readable identifier for the classifier.

        As BERT model training is inherently nondeterministic and this identifier should
        change for each classifier instance with different *behaviour*, only classifiers
        that are not yet fitted have deterministic IDs.

        For fitted classifiers, a random component is generated once during training and
        persisted with the model.
        """

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
        """
        Context manager for safely enabling dropout during inference.

        This ensures the model is always returned to its original state,
        even if an exception occurs during inference.

        Sets a unique random seed for this variant if one was specified during
        variant creation. Also temporarily sets the dropout rate if one was
        specified, then restores the original rates afterwards.
        """
        was_training = self.model.training  # type: ignore[attr-defined]

        dropout_layers = [
            m for m in self.model.modules() if isinstance(m, torch.nn.Dropout)
        ]

        # Log dropout configuration for debugging (only once per variant)
        if not hasattr(self, "_dropout_logged"):
            if not dropout_layers:
                logger.warning(
                    "⚠️  No dropout layers found in model %s. "
                    "Ensemble variants may produce identical predictions.",
                    self.model_name,
                )
            else:
                original_rates = {layer.p for layer in dropout_layers}
                logger.info(
                    "✓ Found %d dropout layers with original rates: %s",
                    len(dropout_layers),
                    original_rates,
                )
            self._dropout_logged = True

        original_dropout_rates = [layer.p for layer in dropout_layers]

        if self._variant_dropout_rate is not None and dropout_layers:
            for layer in dropout_layers:
                layer.p = self._variant_dropout_rate
            logger.debug(
                "Temporarily set dropout rate to %.2f for %d layers",
                self._variant_dropout_rate,
                len(dropout_layers),
            )

        # A unique random seed per variant at inference time ensures different dropout
        # masks across variants
        if self._variant_seed is not None:
            torch.manual_seed(self._variant_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._variant_seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(self._variant_seed)

        self.model.train()  # type: ignore[attr-defined]
        try:
            yield
        finally:
            # Restore original dropout rates
            for layer, original_rate in zip(dropout_layers, original_dropout_rates):
                layer.p = original_rate

            self.model.train(was_training)  # type: ignore[attr-defined]

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        return self._predict_batch([text], threshold=threshold)[0]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """Predict whether the supplied texts contain instances of the concept."""
        positive_probs = self.predict_proba_batch(texts)

        if threshold is not None:
            effective_threshold = threshold
        elif self.prediction_threshold is not None:
            effective_threshold = self.prediction_threshold
        else:
            effective_threshold = 0.5

        now = datetime.now()
        labeller = str(self)
        results = []
        for text, pos_prob in zip(texts, positive_probs):
            text_results = []
            if pos_prob >= effective_threshold:
                span = Span(
                    text=text,
                    concept_id=self.concept.wikibase_id,
                    prediction_probability=pos_prob,
                    start_index=0,
                    end_index=len(text),
                    labellers=[labeller],
                    timestamps=[now],
                )
                text_results.append(span)
            results.append(text_results)

        return results

    def predict_proba_batch(self, texts: Sequence[str]) -> list[float]:
        """
        Return P(class=1) per text, independent of the argmax decision.

        This is needed because `predict` only returns a `Span` for positive examples,
        but probability calibration needs examples from probabilities 0 <= p <= 1.
        """
        if self._use_dropout_during_inference:
            with self._dropout_enabled():
                with torch.no_grad():
                    return self._forward_positive_probs(texts)
        self.model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            return self._forward_positive_probs(texts)

    def _forward_positive_probs(self, texts: Sequence[str]) -> list[float]:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)[:, 1].tolist()

    def get_variant(
        self, random_seed: int | None = None, dropout_rate: float = 0.1
    ) -> Self:
        """
        Get a variant of the classifier for Monte Carlo dropout estimation.

        Creates a new classifier instance with the same trained weights but potentially
        different random state for dropout sampling. This enables Monte Carlo Dropout
        for uncertainty estimation in ensembles.

        Args:
            random_seed: Set at inference time to ensure different dropout masks
                across variants. If not set, uses the containing random state.
            dropout_rate: Dropout probability to use during inference for this variant.
                The dropout rate is temporarily set during inference using the
                _dropout_enabled() context manager. Defaults to 0.1.

        Returns:
            A new classifier instance with dropout enabled during inference.
        """
        variant = self.__class__(
            concept=self.concept,
            model_name=self.model_name,
            unfreeze_layers=self.unfreeze_layers,
        )
        variant.model.load_state_dict(self.model.state_dict())
        variant.device = self.device

        variant._variant_seed = random_seed
        variant._variant_dropout_rate = dropout_rate

        variant._use_dropout_during_inference = True  # noqa: SLF001
        variant.is_fitted = self.is_fitted

        return variant

    def _prepare_dataset(self, labelled_passages: list[LabelledPassage]) -> Dataset:
        """
        Prepare a dataset from labelled passages for training.

        Args:
            labelled_passages: List of labelled passages to prepare dataset from

        Returns:
            Dataset: A HuggingFace dataset ready for training
        """
        texts = [passage.text for passage in labelled_passages]
        labels = [
            1
            if any(
                span.concept_id == self.concept.wikibase_id for span in passage.spans
            )
            else 0
            for passage in labelled_passages
        ]

        # To optimise the speed of the matrix multiplications in our model, we pad all of
        # the passages in each batch to have the same length. This effectively makes every
        # passage in the batch the length of the LONGEST passage in the batch. We have a
        # few verrrrry long passages in our dataset. Matching their length could create a
        # huuuuuge token matrix, leading to memory issues, and breaking our
        # training/inference runs!
        # To mitigate this issue, we enforce a maximum length of 512 tokens for all
        # passages in each batch - we drop any tokens we exceed this limit. Most of the
        # passages in our dataset should be shorter than this limit, but it's worth
        # keeping in mind that we WILL lose some information by truncating those longer
        # passages. This is a trade-off we're willing to make, as the speed of the model's
        # matrix multiplications is more important than the loss of a few tokens. We can
        # also resolve some of this by using a more consistent chunking strategy, but
        # that's out of the scope of this codebase.
        tokenized_inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors=None
        )

        return Dataset.from_dict({**tokenized_inputs, "labels": labels})

    def fit(
        self,
        labelled_passages: list[LabelledPassage],
        validation_size: float = 0.2,
        enable_wandb: bool = False,
        **kwargs,
    ) -> "BertBasedClassifier":
        """
        Fine tune the base model using the labelled passages of the supplied concept.

        The model is fine-tuned using several techniques for faster and more stable
        training:

        - We use a pre-trained model as the base model. By default, we use the
            ModernBERT model from AnswerDotAI, which includes a bunch of smart tricks
            to make it more performant and robust. See https://huggingface.co/blog/modernbert
            for more detail.
        - We freeze the weights of the model's backbone (where most of its fundamental
            language understanding smarts come from) and only adjust the parameters in
            the model's task-specific classification head during training - with the
            default model, that means we're only changing ~0.4% of the total model
            weights! This dramatically reduces the training time and memory usage, while
            still producing a performant classifier.
        - Because we're only training the head, we can use a relatively high learning
            rate and batch size, even on modest hardware.
        - We use a cosine learning rate scheduler, giving us a smooth learning rate
            decay over each epoch.
        - We set a warmup period for the first 6% of the run to stabilise the weights
            in the early stages of training.
        - We set up an early stopping callback to end the training run as soon as we see
            the model's performance improvements slowing down, or beginning to overfit.
        - We set weight decay of 0.1. By adding a small penalty to the weights during
            training, we discourages them from growing too large. Penalising the large
            weights incentiveses the model to learn simpler, more generalisable patterns
            from the training data, making the final model less spiky and more reliable.

        Args:
            labelled_passages: The labelled passages to train the classifier on.
            validation_size: The proportion of labelled passages to use for validation.
            enable_wandb: Whether to enable W&B logging for training metrics and model checkpoints.
            **kwargs: Additional keyword arguments passed to the base class
        Returns:
            BertBasedClassifier: The trained classifier
        """
        if len(labelled_passages) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for "
                f"{self.concept.wikibase_id}. At least 10 are required."
            )

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
        train_dataset = self._prepare_dataset(train_passages)
        validation_dataset = self._prepare_dataset(val_passages)

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
                dataloader_num_workers=0,
                report_to=["wandb"] if enable_wandb else [],
                disable_tqdm=True,
                # W&B-specific settings when enabled
                run_name=f"{self.concept.id}_{self.name}" if enable_wandb else None,
                log_level="info" if enable_wandb else "warning",
            )

            trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                class_weights=class_weights_tensor,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=2, early_stopping_threshold=0
                    )
                ],
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
