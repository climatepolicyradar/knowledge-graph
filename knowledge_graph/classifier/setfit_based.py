import logging
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from rich.logging import RichHandler
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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




class SetfitBasedClassifier(
    Classifier,
    GPUBoundClassifier,
    VariantEnabledClassifier,
    ProbabilityCapableClassifier,
):
    """
    Classifier that uses SetFit for few-shot text classification.

    SetFit is a framework for efficient few-shot text classification using sentence
    transformers. It combines contrastive learning on sentence pairs with a
    classification head, making it particularly effective with small training datasets.

    This classifier uses a pre-trained sentence transformer model, fine-tuned on
    labelled passages to identify instances of a concept. The model is trained to
    produce binary predictions, ie whether a given text contains an instance of the
    concept but not where the concept is specifically mentioned in the given text.

    SetFit advantages:
    - More efficient with small datasets (designed for few-shot learning)
    - Faster training than full transformer fine-tuning
    - Competitive performance with much less data
    """

    def __init__(
        self,
        concept: Concept,
        base_model: str = "BAAI/bge-small-en-v1.5",
    ):
        super().__init__(concept)
        self.base_model = base_model
        self._use_dropout_during_inference = False

        if torch.backends.mps.is_available():
            self.training_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.training_device = torch.device("cuda")
        else:
            self.training_device = torch.device("cpu")

        self.model = SetFitModel.from_pretrained(base_model)
        self.model.to(self.training_device)  # type: ignore

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.base_model,
        )

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict whether the supplied texts contain instances of the concept."""
        # Get predictions and probabilities
        predictions = self.model(texts)  # type: ignore
        probabilities = self.model.predict_proba(texts)  # type: ignore

        results = []
        for text, prediction, proba in zip(texts, predictions, probabilities):
            text_results = []
            # SetFit returns 1 for positive predictions and 0 for negative
            if prediction == 1:
                # proba is [neg_prob, pos_prob], we want the positive probability
                span = Span(
                    text=text,
                    concept_id=self.concept.wikibase_id,
                    prediction_probability=float(proba[1]),
                    start_index=0,
                    end_index=len(text),
                    labellers=[str(self)],
                    timestamps=[datetime.now()],
                )
                text_results.append(span)
            results.append(text_results)

        return results

    def get_variant(self) -> Self:
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

        return Dataset.from_dict({"text": texts, "label": labels})

    def fit(
        self,
        train_validation_data: list[LabelledPassage],
        validation_size: float = 0.2,
        enable_wandb: bool = False,
        **kwargs,
    ) -> "SetfitBasedClassifier":
        """
        Fine tune the base model using the labelled passages of the supplied concept.

        SetFit training uses a two-stage approach:

        1. Contrastive fine-tuning: The sentence transformer is fine-tuned using
           contrastive learning on pairs of sentences. This helps the model learn
           to distinguish between positive and negative examples.

        2. Classification head training: A simple classification head (logistic
           regression) is trained on top of the fine-tuned embeddings.

        This approach is much more efficient than full transformer fine-tuning and
        works particularly well with small datasets.

        Args:
            train_validation_data: The labelled passages to train the classifier on.
            validation_size: The proportion of labelled passages to use for validation.
            enable_wandb: Whether to enable W&B logging for training metrics and model checkpoints.
            **kwargs: Additional keyword arguments passed to the base class
        Returns:
            SetfitBasedClassifier: The trained classifier
        """
        super().fit(**kwargs)

        if len(train_validation_data) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for "
                f"{self.concept.wikibase_id}. At least 10 are required."
            )

        labels = [
            1
            if any(span.concept_id == self.concept.wikibase_id for span in p.spans)
            else 0
            for p in train_validation_data
        ]

        # Split passages into training and validation sets. Stratify to maintain the
        # distribution of positive and negative passages.
        train_passages, val_passages, _, _ = train_test_split(
            train_validation_data,
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
                    (sum(train_dataset["label"]) / len(train_dataset)) * 100,
                    (sum(validation_dataset["label"]) / len(validation_dataset)) * 100,
                ],
            }
        )

        logger.info(
            pd.DataFrame(stats_df)
            .round(2)
            .to_markdown(index=False, tablefmt="rounded_grid")
        )

        # SetFit training configuration using the updated API
        # Instead of using samples_per_label which can generate too many pairs,
        # use max_steps to directly limit the number of training steps.
        # Target: ~300-500 steps regardless of dataset size (takes 5-10 minutes)
        dataset_size = len(train_dataset)

        # Calculate max_steps based on dataset size
        # Small datasets: more steps per example
        # Large datasets: fewer steps total
        if dataset_size < 50:
            max_steps = 500
        elif dataset_size < 200:
            max_steps = 400
        else:
            max_steps = 300

        logger.info(
            f"Training with max_steps={max_steps} for dataset with {dataset_size} examples"
        )

        args = TrainingArguments(
            batch_size=8,
            num_epochs=1,
            max_steps=max_steps,  # Directly limit training steps
            eval_strategy="steps",
            eval_steps=max_steps,  # Evaluate at the end
            save_strategy="steps",
            save_steps=max_steps,  # Save at the end
            load_best_model_at_end=False,  # Only one checkpoint
            samples_per_label=2,  # Use default, but limited by max_steps
            body_learning_rate=2e-5,
            head_learning_rate=2e-2,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            metric="accuracy",
            column_mapping={
                "text": "text",
                "label": "label",
            },  # Map dataset columns to text/label expected by trainer
        )

        logger.info("🚀 Starting training...")
        trainer.train()

        logger.info("🎯 Evaluating on validation set...")
        # Get predictions on validation set
        val_texts = validation_dataset["text"]
        val_labels = validation_dataset["label"]
        val_predictions = self.model(val_texts)  # type: ignore

        # Compute metrics manually
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_predictions, average="binary"
        )
        accuracy = accuracy_score(val_labels, val_predictions)

        results_df = pd.DataFrame(
            {
                "Metric": [
                    "F1 Score",
                    "Accuracy",
                    "Precision",
                    "Recall",
                ],
                "Value": [
                    float(f1),
                    float(accuracy),
                    float(precision),
                    float(recall),
                ],
            }
        )
        logger.info("Final Validation Results")
        logger.info(
            results_df.round(4).to_markdown(index=False, tablefmt="rounded_grid")
        )

        logger.info("📊 Final F1 score: %.4f", float(f1))

        logger.info("✅ Training complete for concept %s!", self.concept.id)
        return self