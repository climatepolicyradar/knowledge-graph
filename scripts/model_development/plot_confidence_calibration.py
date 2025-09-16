from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console

from knowledge_graph.classifier import (
    Classifier,
)
from knowledge_graph.classifier.classifier import ProbabilityCapableClassifier
from knowledge_graph.classifier.ensemble import VotingLLMClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.config import metrics_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.wikibase import WikibaseSession
from scripts.model_development.local_inference_helpers import (
    label_passages,
    save_labelled_passages_and_classifier,
)

app = typer.Typer()
console = Console(highlight=False)

# These concepts were chosen as representative concepts with well-defined concept-store
# entries in July 2025.
# https://linear.app/climate-policy-radar/issue/SCI-434/identify-5-representative-concepts-with-well-defined-concept-store
CONCEPT_IDS = [
    "Q1285",  # ban
    # "Q913", # restorative justice
    # "Q760", # extractive sector
    # "Q715", # tax
]


def get_classifiers_and_inference_settings(
    concept: Concept,
) -> list[tuple[Classifier, dict[str, Any], int]]:
    """
    Get classifiers for a concept to run confidence calibration on.

    Also returns any kwargs to pass to the model's predict method and batch size per
    classifier.

    :returns list[tuple[Classifier, dict[str, Any], int]]: list of tuples of classifier,
    kwargs to pass to the predict method, and batch size
    """

    # TODO: load appropriate secrets for classifiers

    voting_classifier_predict_passage_kwargs = {"passage_level": True}

    return [
        (
            VotingLLMClassifier(
                concept=concept, model_name="gpt-4o-mini", n_classifiers=10
            ),
            voting_classifier_predict_passage_kwargs,
            20,
        ),
        (
            VotingLLMClassifier(concept=concept, model_name="gpt-4o", n_classifiers=10),
            voting_classifier_predict_passage_kwargs,
            20,
        ),
        (
            VotingLLMClassifier(
                concept=concept,
                model_name="claude-3-5-sonnet-20241022",
                n_classifiers=10,
            ),
            voting_classifier_predict_passage_kwargs,
            20,
        ),
    ]


def extract_passage_level_data(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
) -> tuple[list[float], list[bool]]:
    """
    Extract passage-level predicted probabilities and human labels.

    These should be for a model that predicted one span per passage only.

    :returns tuple[list[float], list[bool]]: corresponding prediction probabilities
    and human labels (boolean) for all passages which have a model label
    """

    if max([len(passage.spans) for passage in model_labelled_passages]) > 1:
        raise ValueError(
            "Model labelled passages have been passed to `extract_passage_level_data` which have more than one span label per passage. Only passage-level labels should be used."
        )

    if not all(
        isinstance(span.prediction_probability, float)
        for passage in model_labelled_passages
        for span in passage.spans
    ):
        raise ValueError("Some model labels were passed without probabilities.")

    predicted_probs = []
    human_labels = []

    passage_id_to_human_label = {
        passage.id: len(passage.spans) > 0 for passage in human_labelled_passages
    }

    for model_passage in model_labelled_passages:
        if not (human_label := passage_id_to_human_label.get(model_passage.id)):
            continue

        if model_passage.spans:
            predicted_prob = model_passage.spans[0].prediction_probability
            predicted_probs.append(predicted_prob)
            human_labels.append(human_label)

    return predicted_probs, human_labels


def calculate_calibration_stats(
    predicted_probs: list[float],
    human_labels: list[bool],
    n_bins: int = 10,
):
    """
    Calculate statistics used for calibration calculations.

    These are:
    - bin_boundaries: (n_bins+1) equally spaced numbers between 0 and 1
    - bin_accuracies: the proportion of positive human labels per bin
    - bin_confidences: the average confidence of model predictions between the bin
        boundaries
    - bin_counts: the number of model predictions with confidence in the bin boundaries
    """

    predicted_probs_array = np.array(predicted_probs)
    human_labels_array = np.array(human_labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs_array > bin_lower) & (
            predicted_probs_array <= bin_upper
        )
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = human_labels_array[in_bin].mean()
            avg_confidence_in_bin = predicted_probs_array[in_bin].mean()
            count_in_bin = in_bin.sum()
        else:
            accuracy_in_bin = 0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
            count_in_bin = 0

        bin_accuracies.append(accuracy_in_bin)
        bin_confidences.append(avg_confidence_in_bin)
        bin_counts.append(count_in_bin)

    return bin_boundaries, bin_accuracies, bin_confidences, bin_counts


def calculate_expected_calibration_error(
    bin_accuracies: list[float],
    bin_confidences: list[float],
    bin_counts: list[int],
) -> float:
    """
    Calculate expected calibration error.

    This is the sum of (predicted_confidence - proportion_true_labels) weighted by the
    width of the bins. We assume uniform length bins here, so it's just the sum.
    """
    if len(bin_accuracies) == 0:
        return 0.0

    accuracies = np.asarray(bin_accuracies, dtype=float)
    confidences = np.asarray(bin_confidences, dtype=float)
    counts = np.asarray(bin_counts, dtype=float)

    total = counts.sum()
    if total == 0:
        return 0.0

    weighted_abs_diff = (counts / total) * np.abs(accuracies - confidences)
    return float(weighted_abs_diff.sum())


def plot_confidence_calibration(
    predicted_probs: list[float],
    human_labels: list[bool],
    concept: Concept,
    classifier_name: str,
    n_bins: int = 10,
) -> None:
    """Create confidence calibration plots for a classifier."""

    bin_boundaries, bin_accuracies, bin_confidences, bin_counts = (
        calculate_calibration_stats(predicted_probs, human_labels, n_bins)
    )
    ece = calculate_expected_calibration_error(
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )

    fig, (ax_calibration, ax_histogram) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{classifier_name}\nConcept: {concept.preferred_label}", y=0.98, wrap=True
    )

    ax_calibration.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax_calibration.plot(bin_confidences, bin_accuracies, "bo-", label="Classifier")
    ax_calibration.set_xlabel("Mean Predicted Probability")
    ax_calibration.set_ylabel("Fraction of Positives")
    ax_calibration.set_title(
        f"Confidence Calibration. ECE = {ece}",
        wrap=True,
    )
    ax_calibration.legend()
    ax_calibration.grid(True, alpha=0.3)

    ax_histogram.hist(
        predicted_probs,
        bins=bin_boundaries.tolist(),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax_histogram.set_xlabel("Predicted Probability")
    ax_histogram.set_ylabel("Count")
    ax_histogram.set_title(
        "Probability Distribution",
        wrap=True,
    )
    ax_histogram.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    safe_classifier_name = classifier_name.replace("/", "_").replace(" ", "_")
    safe_concept_name = concept.preferred_label.replace("/", "_").replace(" ", "_")
    filename = f"confidence_calibration_{safe_concept_name}_{safe_classifier_name}.png"

    metrics_dir.mkdir(exist_ok=True)
    output_path = metrics_dir / filename

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    console.log(f"    Saved plot: {output_path}")


@app.command()
def main(passage_limit: Optional[int] = None):
    """
    Plots confidence calibration curves for a set of probability-capable classifiers.

    The classifiers should not require fitting, and should be able to predict
    probabilities.

    Note:
        Intermediate results and the final classifiers are saved to standardised paths
    """
    argilla = ArgillaSession()
    wikibase = WikibaseSession()

    concepts: list[Concept] = []

    for concept_id in CONCEPT_IDS:
        concept = wikibase.get_concept(WikibaseID(concept_id))
        concept.labelled_passages = argilla.pull_labelled_passages(concept)
        if len(concept.labelled_passages) == 0:
            console.log(
                f"🚫 No passages found in Argilla for concept {concept_id}. Concept will be excluded from calibration measurement."
            )
            continue

        if passage_limit is not None:
            concept.labelled_passages = concept.labelled_passages[:passage_limit]

        console.log(
            f"🧠 Loaded concept [bold white]{concept}[/bold white] with {len(concept.labelled_passages)} labelled passages."
        )
        concepts.append(concept)

    console.log(f"Loaded {len(concepts)}/{len(CONCEPT_IDS)} concepts successfully.")

    classifiers_by_concept: dict[Concept, list[tuple[Classifier, dict, int]]] = {
        concept: get_classifiers_and_inference_settings(concept) for concept in concepts
    }

    first_concept_classifiers = [i[0] for i in list(classifiers_by_concept.values())[0]]

    if probability_incapable_classifiers := [
        clf
        for clf in first_concept_classifiers
        if not isinstance(clf, ProbabilityCapableClassifier)
    ]:
        raise ValueError(
            f"All classifiers used in this script must output probabilities.\nThe following classifiers specified don't: {probability_incapable_classifiers}"
        )

    console.log(
        f"Running inference on passages for each concept for classifiers:\n {'|'.join(str(c) for c in first_concept_classifiers)}"
    )

    console.log("Labelling passages with classifiers:")
    classifier_labelled_passages_by_concept = defaultdict(list)
    for concept, classifier_and_batch_size in classifiers_by_concept.items():
        console.log(f"Concept {concept}")
        for classifier, predict_kwargs, batch_size in classifier_and_batch_size:
            labelled_passages = label_passages(
                concept.labelled_passages,
                classifier=classifier,
                batch_size=batch_size,
                predict_kwargs=predict_kwargs,
            )
            classifier_labelled_passages_by_concept[concept].append(labelled_passages)
            save_labelled_passages_and_classifier(
                labelled_passages=labelled_passages,
                classifier=classifier,
            )

    console.log("Creating confidence calibration plots...")

    for concept in concepts:
        console.log(f"Processing concept: {concept}")

        human_labelled_passages = concept.labelled_passages
        classifier_results = classifier_labelled_passages_by_concept[concept]

        for classifier_idx, model_predictions in enumerate(classifier_results):
            classifier_name = str(classifiers_by_concept[concept][classifier_idx][0])
            console.log(f"  Plotting calibration for: {classifier_name}")

            predicted_probs, human_labels = extract_passage_level_data(
                human_labelled_passages, model_predictions
            )

            if len(predicted_probs) == 0:
                console.log(f"    No predictions found for {classifier_name}")
                continue

            plot_confidence_calibration(
                predicted_probs=predicted_probs,
                human_labels=human_labels,
                concept=concept,
                classifier_name=classifier_name,
                n_bins=10,
            )


if __name__ == "__main__":
    app()
