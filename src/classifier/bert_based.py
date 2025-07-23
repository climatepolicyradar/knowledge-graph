import os
import tempfile
from datetime import datetime

import torch  # type: ignore[import-untyped]
from datasets import Dataset
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.classifier.classifier import Classifier, GPUBoundClassifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.span import Span


class BertBasedClassifier(Classifier, GPUBoundClassifier):
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
        base_model: str = "climatebert/distilroberta-base-climate-f",
    ):
        super().__init__(concept)
        self.base_model = base_model

        # For training, we can use GPU/MPS if available
        if torch.backends.mps.is_available():
            self.training_device = torch.device("mps")
        elif torch.cuda.is_available():
            self.training_device = torch.device("cuda")
        else:
            self.training_device = torch.device("cpu")

        # Initialize model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

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
        self.model.to(self.training_device)

    def __repr__(self):
        """Return a string representation of the classifier."""
        return f'{self.name}({self.concept.preferred_label}, base_model="{self.base_model}")'

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict whether the supplied texts contain instances of the concept."""
        self.model.eval()

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
                    start_index=0,
                    end_index=len(text),
                    labellers=[str(self)],
                    timestamps=[datetime.now()],
                )
                text_results.append(span)
            results.append(text_results)

        return results

    def _prepare_dataset(self, labelled_passages: list[LabelledPassage]) -> Dataset:
        """
        Prepare a dataset from labelled passages for training.

        Args:
            labelled_passages: List of labelled passages to prepare dataset from

        Returns:
            Dataset: A HuggingFace dataset ready for training
        """
        texts = []
        labels = []

        for passage in labelled_passages:
            contains_concept = any(
                span.concept_id == self.concept.wikibase_id for span in passage.spans
            )
            texts.append(passage.text)
            labels.append(1 if contains_concept else 0)

        tokenized_inputs = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        )

        return Dataset.from_dict({**tokenized_inputs, "labels": labels})

    def fit(
        self,
        *,
        use_wandb: bool = False,
        **kwargs,
    ) -> "BertBasedClassifier":
        """
        Fine tune the base model using the labelled passages of the supplied concept.

        The model will be trained to predict whether a supplied text contains an
        instance of the concept.

        Args:
            use_wandb: Whether to use Weights & Biases for logging
            **kwargs: Additional keyword arguments passed to the base class

        Returns:
            BertBasedClassifier: The trained classifier
        """
        super().fit(**kwargs)

        if len(self.concept.labelled_passages) < 10:
            raise ValueError(
                f"Not enough labelled passages to train a {self.name} for {self.concept.wikibase_id}"
            )

        tokenized_dataset = self._prepare_dataset(self.concept.labelled_passages)

        with tempfile.TemporaryDirectory() as temp_dir:
            # N.B. These training arguments should probably be configurable. Deliberately
            # keeping them frozen for the time being while developing the overall flow.
            training_args = TrainingArguments(
                output_dir=os.path.join(temp_dir, "results"),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                learning_rate=5e-5,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                logging_dir=os.path.join(temp_dir, "logs"),
                logging_steps=10,
                report_to="wandb" if use_wandb else "none",
                disable_tqdm=True,
            )

            trainer = Trainer(
                model=self.model, args=training_args, train_dataset=tokenized_dataset
            )
            trainer.train()

        return self
