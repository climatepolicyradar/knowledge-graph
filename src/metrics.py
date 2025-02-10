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

    def support(self) -> int:
        """Total number of samples in the confusion matrix"""
        return (
            self.true_positives
            + self.false_negatives
            + self.false_positives
            + self.true_negatives
        )

    def has_any_positives(self) -> bool:
        """Check if there are any positive cases (either true or predicted)"""
        return (
            self.true_positives > 0
            or self.false_positives > 0
            or self.false_negatives > 0
        )

    def precision(self) -> float:
        """https://en.wikipedia.org/wiki/Precision_and_recall"""
        # If there are no positive cases at all (true or predicted), that's perfect precision
        if not self.has_any_positives():
            return 1.0
        try:
            return self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return 0

    def recall(self) -> float:
        """https://en.wikipedia.org/wiki/Precision_and_recall"""
        # If there are no positive cases at all (true or predicted), that's perfect recall
        if not self.has_any_positives():
            return 1.0
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
        # If there are no positive cases at all (true or predicted), that's perfect F1
        if not self.has_any_positives():
            return 1.0
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
    ground_truth_passages: list[LabelledPassage],
    predicted_passages: list[LabelledPassage],
    threshold: float,
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
) -> ConfusionMatrix:
    """
    Count the passage-level metrics for a given set of human and model labelled passages.

    A passage is considered a true negative if both ground truth and prediction have
    exactly matching text and no spans.

    :param list[LabelledPassage] ground_truth_passages: A set of gold-standard spans
    :param list[LabelledPassage] predicted_passages: A set of predicted spans
    :return ConfusionMatrix: The resulting confusion matrix for calculating
    passage-level metrics
    """
    cm = ConfusionMatrix()

    # First check if this is an all-negative case
    has_any_ground_truth_spans = any(len(p.spans) > 0 for p in ground_truth_passages)
    has_any_predicted_spans = any(len(p.spans) > 0 for p in predicted_passages)

    # If neither has any spans, this is a perfect match - all true negatives
    if not has_any_ground_truth_spans and not has_any_predicted_spans:
        cm.true_negatives = len(ground_truth_passages)
        return cm

    # If ground truth has no spans but predictions do, all false positives
    if not has_any_ground_truth_spans and has_any_predicted_spans:
        cm.false_positives = sum(1 for p in predicted_passages if len(p.spans) > 0)
        cm.true_negatives = len(predicted_passages) - cm.false_positives
        return cm

    # If ground truth has spans but predictions don't, all false negatives
    if has_any_ground_truth_spans and not has_any_predicted_spans:
        cm.false_negatives = sum(1 for p in ground_truth_passages if len(p.spans) > 0)
        cm.true_negatives = len(ground_truth_passages) - cm.false_negatives
        return cm

    # Otherwise, do the normal passage-by-passage comparison
    for gt_passage, pred_passage in zip(ground_truth_passages, predicted_passages):
        gt_has_spans = len(gt_passage.spans) > 0
        pred_has_spans = len(pred_passage.spans) > 0

        if gt_has_spans and pred_has_spans:
            cm.true_positives += 1
        elif not gt_has_spans and not pred_has_spans:
            cm.true_negatives += 1
        elif gt_has_spans and not pred_has_spans:
            cm.false_negatives += 1
        else:  # not gt_has_spans and pred_has_spans
            cm.false_positives += 1

    return cm
