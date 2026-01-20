from dataclasses import dataclass

from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import jaccard_similarity


@dataclass
class ConfusionMatrix:
    """A class to represent a confusion matrix"""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def support(self) -> int:
        """Total number of samples in the confusion matrix"""
        return (
            self.true_positives
            + self.false_negatives
            + self.false_positives
            + self.true_negatives
        )

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

        return self.f_beta_score(beta=1.0)

    def f_beta_score(self, beta: float = 1.0) -> float:
        """https://en.wikipedia.org/wiki/F-score#F%CE%B2_score"""

        if self.true_positives == 0:
            return 0.0

        if beta <= 0:
            raise ValueError(
                "beta must be positive (e.g., 0.5 for F0.5, 1.0 for F1, 2.0 for F2)"
            )

        precision = self.precision()
        recall = self.recall()

        if precision == 0 and recall == 0:
            return 0.0

        beta_squared = beta**2
        numerator = (1 + beta_squared) * (precision * recall)
        denominator = (beta_squared * precision) + recall

        return numerator / denominator

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
    ground_truth_passages: list[LabelledPassage],
    predicted_passages: list[LabelledPassage],
    threshold: float = 0.9,
) -> ConfusionMatrix:
    """
    Count the span-level metrics for a given set of human and model labelled passages

    :param list[LabelledPassage] ground_truth_passages: A set of gold-standard spans
    :param list[LabelledPassage] predicted_passages: A set of predicted spans
    :param float threshold: The Jaccard similarity threshold to consider overlapping
    spans as a match
    :return ConfusionMatrix: The resulting confusion matrix for calculating span-level
    metrics
    """
    cm = ConfusionMatrix()

    for ground_truth_passage, predicted_passage in zip(
        ground_truth_passages, predicted_passages
    ):
        # If both passages have no spans, count as true negative
        if not ground_truth_passage.spans and not predicted_passage.spans:
            cm.true_negatives += 1
            continue

        for ground_truth_span in ground_truth_passage.spans:
            found = False
            for predicted_span in predicted_passage.spans:
                if jaccard_similarity(ground_truth_span, predicted_span) > threshold:
                    found = True
                    cm.true_positives += 1
                    break
            if not found:
                cm.false_negatives += 1
                break

        for predicted_span in predicted_passage.spans:
            found = False
            for ground_truth_span in ground_truth_passage.spans:
                if jaccard_similarity(predicted_span, ground_truth_span) > threshold:
                    found = True
                    break
            if not found:
                cm.false_positives += 1
                break

    return cm


def count_passage_level_metrics(
    ground_truth_passages: list[LabelledPassage],
    predicted_passages: list[LabelledPassage],
    threshold: float = 0.9,
) -> ConfusionMatrix:
    """
    Count the passage-level metrics for a given set of human and model labelled passages

    :param list[LabelledPassage] ground_truth_passages: A set of gold-standard spans
    :param list[LabelledPassage] predicted_passages: A set of predicted spans
    :return ConfusionMatrix: The resulting confusion matrix for calculating
    passage-level metrics
    """
    cm = ConfusionMatrix()

    ground_truth_positive_passages = set(
        passage.id for passage in ground_truth_passages if len(passage.spans) > 0
    )
    predicted_positive_passages = set(
        passage.id for passage in predicted_passages if len(passage.spans) > 0
    )
    ground_truth_negative_passages = set(
        passage.id for passage in ground_truth_passages if len(passage.spans) == 0
    )
    predicted_negative_passages = set(
        passage.id for passage in predicted_passages if len(passage.spans) == 0
    )

    cm.true_positives = len(
        ground_truth_positive_passages & predicted_positive_passages
    )
    cm.false_positives = len(
        predicted_positive_passages - ground_truth_positive_passages
    )
    cm.true_negatives = len(
        ground_truth_negative_passages & predicted_negative_passages
    )
    cm.false_negatives = len(
        ground_truth_positive_passages - predicted_positive_passages
    )

    return cm
