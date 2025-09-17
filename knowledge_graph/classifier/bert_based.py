import logging
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from rich.logging import RichHandler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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

from knowledge_graph.classifier.classifier import Classifier, GPUBoundClassifier
from knowledge_graph.classifier.uncertainty_mixin import UncertaintyMixin
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


class BertBasedClassifier(Classifier, GPUBoundClassifier, UncertaintyMixin):
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
        base_model: str = "answerdotai/ModernBERT-base",
    ):
        super().__init__(concept)
        self.base_model = base_model
        self._use_dropout_during_inference = False

        # For training, we can use GPU/MPS if available
        if torch.backends.mps.is_available():
            self.training_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.training_device = torch.device("cuda")
        else:
            self.training_device = torch.device("cpu")

        # Initialize model and tokenizer
        self.model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(base_model)
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model)

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
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.base_model,
        )

    @contextmanager
    def _dropout_enabled(self):
        """
        Context manager for safely enabling dropout during inference.

        This ensures the model is always returned to its original state,
        even if an exception occurs during uncertainty estimation.
        """
        was_training = self.model.training  # type: ignore[attr-defined]
        self.model.train()  # type: ignore[attr-defined]
        try:
            yield
        finally:
            self.model.train(was_training)  # type: ignore[attr-defined]

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict whether the supplied texts contain instances of the concept."""

        if getattr(self, "_use_dropout_during_inference", False):
            with self._dropout_enabled():
                predictions = self.pipeline(texts, padding=True, truncation=True)
        else:
            self.model.eval()  # type: ignore[attr-defined]
            predictions = self.pipeline(texts, padding=True, truncation=True)

        results = []
        for text, prediction in zip(texts, predictions):
            text_results = []
            # By default, the huggingface text classification pipeline returns LABEL_0
            # for negative predictions and LABEL_1 for positive predictions. We check
            # for LABEL_1 to determine if the text contains an instance of the concept.
            if prediction["label"] == "LABEL_1":
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

    def get_variant_sub_classifier(self) -> Self:
        """Get a variant of the classifier for Monte Carlo dropout estimation."""
        variant = deepcopy(self)
        variant._use_dropout_during_inference = True  # noqa: SLF001
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

        tokenized_inputs = self.tokenizer(  # type: ignore[operator]
            texts, padding=True, truncation=True, max_length=512, return_tensors=None
        )

        return Dataset.from_dict({**tokenized_inputs, "labels": labels})

    def fit(
        self,
        validation_size: float = 0.2,
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
            rate (5e-4) and batch size (64), even on modest hardware.
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
            validation_size: The proportion of labelled passages to use for validation.
            **kwargs: Additional keyword arguments passed to the base class
        Returns:
            BertBasedClassifier: The trained classifier
        """
        super().fit(**kwargs)

        if len(self.concept.labelled_passages) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for "
                f"{self.concept.wikibase_id}. At least 10 are required."
            )

        passages = self.concept.labelled_passages
        labels = [
            1
            if any(span.concept_id == self.concept.wikibase_id for span in p.spans)
            else 0
            for p in passages
        ]

        # Split passages into training and validation sets. Stratify to maintain the
        # distribution of positive and negative passages.
        train_passages, val_passages, _, _ = train_test_split(
            passages,
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

        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=os.path.join(temp_dir, "results"),
                num_train_epochs=3,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                learning_rate=5e-4,
                weight_decay=0.01,
                warmup_ratio=0.06,
                lr_scheduler_type="cosine",
                optim="adamw_torch_fused",
                fp16=False,
                dataloader_pin_memory=False,
                logging_dir=os.path.join(temp_dir, "logs"),
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=100,
                save_steps=200,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                dataloader_num_workers=2,
                report_to=[],
                disable_tqdm=True,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=3, early_stopping_threshold=0.001
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
        return self
