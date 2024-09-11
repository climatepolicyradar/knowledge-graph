from dataclasses import dataclass

from src.labelled_passage import LabelledPassage
from src.span import jaccard_similarity


@dataclass
class ConfusionMatrix:
    """A class to represent a confusion matrix"""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def precision(self) -> float:
        """https://en.wikipedia.org/wiki/Precision_and_recall"""
        try:
            return self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return 0

    def recall(self) -> float:
        """https://en.wikipedia.org/wiki/Precision_and_recall"""
        try:
            return self.true_positives / (self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            return 0

    def accuracy(self) -> float:
        """https://en.wikipedia.org/wiki/Accuracy_and_precision"""
        try:
            return (self.true_positives + self.true_negatives) / (
                self.true_positives
                + self.true_negatives
                + self.false_positives
                + self.false_negatives
            )
        except ZeroDivisionError:
            return 0

    def f1_score(self) -> float:
        """https://en.wikipedia.org/wiki/F-score"""
        try:
            return (2 * self.true_positives) / (
                2 * self.true_positives + self.false_positives + self.false_negatives
            )
        except ZeroDivisionError:
            return 0

    def cohens_kappa(self) -> float:
        """https://en.wikipedia.org/wiki/Cohen%27s_kappa"""
        try:
            total = (
                self.true_positives
                + self.true_negatives
                + self.false_positives
                + self.false_negatives
            )
            observed_agreement = (self.true_positives + self.true_negatives) / total
            expected_agreement = (
                (self.true_positives + self.false_positives)
                * (self.true_positives + self.false_negatives)
                + (self.false_positives + self.true_negatives)
                * (self.false_negatives + self.true_negatives)
            ) / (total**2)
            return (observed_agreement - expected_agreement) / (1 - expected_agreement)
        except ZeroDivisionError:
            return 0


def count_span_level_metrics(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
    threshold: float,
) -> ConfusionMatrix:
    """
    Count the span-level metrics for a given set of human and model labelled passages

    :param list[LabelledPassage] human_labelled_passages: A set of gold-standard spans
    :param list[LabelledPassage] model_labelled_passages: A set of predicted spans
    :param float threshold: The Jaccard similarity threshold to consider overlapping
    spans as a match
    :return ConfusionMatrix: The resulting confusion matrix for calculating span-level
    metrics
    """
    cm = ConfusionMatrix()

    for human_labelled_passage, model_labelled_passage in zip(
        human_labelled_passages, model_labelled_passages
    ):
        for human_span in human_labelled_passage.spans:
            found = False
            for model_span in model_labelled_passage.spans:
                if jaccard_similarity(human_span, model_span) >= threshold:
                    found = True
                    cm.true_positives += 1
                    break
            if not found:
                cm.false_negatives += 1
                break

        for model_span in model_labelled_passage.spans:
            found = False
            for human_span in human_labelled_passage.spans:
                if jaccard_similarity(model_span, human_span) >= threshold:
                    found = True
                    break
            if not found:
                cm.false_positives += 1
                break

        human_labelled_negative_passages = set(
            passage.id for passage in human_labelled_passages if len(passage.spans) == 0
        )
        model_labelled_negative_passages = set(
            passage.id for passage in model_labelled_passages if len(passage.spans) == 0
        )
        cm.true_negatives += len(
            human_labelled_negative_passages & model_labelled_negative_passages
        )

    return cm


def count_passage_level_metrics(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
) -> ConfusionMatrix:
    """
    Count the passage-level metrics for a given set of human and model labelled passages

    :param list[LabelledPassage] human_labelled_passages: A set of gold-standard spans
    :param list[LabelledPassage] model_labelled_passages: A set of predicted spans
    :return ConfusionMatrix: The resulting confusion matrix for calculating
    passage-level metrics
    """
    cm = ConfusionMatrix()

    human_labelled_positive_passages = set(
        passage.id for passage in human_labelled_passages if len(passage.spans) > 0
    )
    model_labelled_positive_passages = set(
        passage.id for passage in model_labelled_passages if len(passage.spans) > 0
    )
    human_labelled_negative_passages = set(
        passage.id for passage in human_labelled_passages if len(passage.spans) == 0
    )
    model_labelled_negative_passages = set(
        passage.id for passage in model_labelled_passages if len(passage.spans) == 0
    )

    cm.true_positives = len(
        human_labelled_positive_passages & model_labelled_positive_passages
    )
    cm.false_positives = len(
        model_labelled_positive_passages - human_labelled_positive_passages
    )
    cm.true_negatives = len(
        human_labelled_negative_passages & model_labelled_negative_passages
    )
    cm.false_negatives = len(
        human_labelled_positive_passages - model_labelled_positive_passages
    )

    return cm
