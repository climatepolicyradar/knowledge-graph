import logging
import os
import random
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
)
from transformers.pipelines import pipeline
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
    ):
        super().__init__(concept)
        self.model_name = model_name

        # Private properties for creating and running inference on classifier variants
        self._use_dropout_during_inference = False
        self._variant_seed = False
        self._variant_dropout_rate = False

        # Random component for nondeterministic ID, generated once when classifier is fitted
        self._random_id_component: str = ""

        # For training, we can use GPU/MPS if available
        if torch.backends.mps.is_available():
            self.training_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.training_device = torch.device("cuda")
        else:
            self.training_device = torch.device("cpu")

        # Initialize model and tokenizer
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        # Always use CPU for inference, to ensure consistency across different deployment
        # environments. Models may be developed on machines with GPU/MPS but need to run
        # reliably on CPU in production pipelines.
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.training_device,
        )

        # Move model to training device (only used during training)
        self.model.to(self.training_device)  # type: ignore

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

        if self._use_dropout_during_inference:
            with self._dropout_enabled():
                predictions = self.pipeline(list(texts), padding=True, truncation=True)
        else:
            self.model.eval()  # type: ignore[attr-defined]
            predictions = self.pipeline(list(texts), padding=True, truncation=True)

        # Use the provided threshold, or fall back to the classifier's default threshold
        effective_threshold = (
            threshold if threshold is not None else self.prediction_threshold
        )

        results = []
        for text, prediction in zip(texts, predictions):
            text_results = []
            # By default, the huggingface text classification pipeline returns LABEL_0
            # for negative predictions and LABEL_1 for positive predictions. We check
            # for LABEL_1 to determine if the text contains an instance of the concept.
            if prediction["label"] == "LABEL_1":
                if (
                    effective_threshold is None
                    or prediction["score"] >= effective_threshold
                ):
                    span = Span(
                        text=text,
                        concept_id=self.concept.wikibase_id,
                        prediction_probability=prediction["score"],
                        start_index=0,
                        end_index=len(text),
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )
                    text_results.append(span)
            results.append(text_results)

        return results

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
        variant = self.__class__(concept=self.concept, model_name=self.model_name)
        variant.model.load_state_dict(self.model.state_dict())
        variant.pipeline = pipeline(
            "text-classification",
            model=variant.model,
            tokenizer=variant.tokenizer,
            device=variant.training_device,
        )

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

        logger.info(
            "Freezing base model. The model's prediction head and classifier will be trained."
        )
        # More efficient parameter freezing - single pass through parameters
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
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
            self.training_device
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
                learning_rate=5e-4,
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
