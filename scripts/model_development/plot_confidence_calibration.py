from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console

from knowledge_graph.classifier import (
    Classifier,
)
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


def get_classifiers_and_batch_sizes(concept: Concept) -> list[tuple[Classifier, int]]:
    """Get classifiers for a concept to run confidence calibration on."""

    return [
        (
            VotingLLMClassifier(
                concept=concept, model_name="gpt-4o-mini", n_classifiers=10
            ),
            20,
        ),
        # (VotingLLMClassifier(concept=concept, model_name="gpt-4o", n_classifiers=10), 20),
        # (VotingLLMClassifier(concept=concept, model_name="claude-3-5-sonnet-20241022", n_classifiers=10), 20),
    ]


def extract_passage_level_data(
    human_labelled_passages: list[LabelledPassage],
    model_predictions: list[LabelledPassage],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract passage-level predicted probabilities and true labels.

    For each passage:
    - Probability = max probability from any span predicted by model
    - True label = 1 if human annotator found any relevant spans, 0 otherwise
    """
    predicted_probs = []
    true_labels = []

    passage_id_to_human_label = {
        passage.id: len(passage.spans) > 0 for passage in human_labelled_passages
    }

    for model_passage in model_predictions:
        if model_passage.spans:
            max_prob = max(
                span.prediction_probability or 0.0 for span in model_passage.spans
            )
        else:
            max_prob = 0.0

        human_label = passage_id_to_human_label.get(model_passage.id, False)

        predicted_probs.append(max_prob)
        true_labels.append(1 if human_label else 0)

    return np.array(predicted_probs), np.array(true_labels)


def plot_confidence_calibration(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    concept: Concept,
    classifier_name: str,
    n_bins: int = 10,
) -> None:
    """Create confidence calibration plots for a classifier."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predicted_probs[in_bin].mean()
            count_in_bin = in_bin.sum()
        else:
            accuracy_in_bin = 0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2
            count_in_bin = 0

        bin_centers.append((bin_lower + bin_upper) / 2)
        bin_accuracies.append(accuracy_in_bin)
        bin_confidences.append(avg_confidence_in_bin)
        bin_counts.append(count_in_bin)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    plt.plot(bin_confidences, bin_accuracies, "bo-", label="Classifier")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Confidence Calibration\n{concept.preferred_label} - {classifier_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(
        predicted_probs,
        bins=bin_boundaries.tolist(),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(
        f"Probability Distribution\n{concept.preferred_label} - {classifier_name}"
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

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

    # TODO: load appropriate secrets for classifiers

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

    classifiers_by_concept: dict[Concept, list[tuple[Classifier, int]]] = {
        concept: get_classifiers_and_batch_sizes(concept) for concept in concepts
    }

    first_concept_classifiers = list(classifiers_by_concept.values())[0]
    console.log(
        f"Running inference on passages for each concept for classifiers:\n {'|'.join(str(c) for c in first_concept_classifiers)}"
    )

    console.log("Labelling passages with classifiers:")
    classifier_labelled_passages_by_concept = defaultdict(list)
    for concept, classifier_and_batch_size in classifiers_by_concept.items():
        console.log(f"Concept {concept}")
        for classifier, batch_size in classifier_and_batch_size:
            labelled_passages = label_passages(
                concept.labelled_passages,
                classifier=classifier,
                batch_size=batch_size,
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

            predicted_probs, true_labels = extract_passage_level_data(
                human_labelled_passages, model_predictions
            )

            if len(predicted_probs) == 0:
                console.log(f"    No predictions found for {classifier_name}")
                continue

            plot_confidence_calibration(
                predicted_probs=predicted_probs,
                true_labels=true_labels,
                concept=concept,
                classifier_name=classifier_name,
                n_bins=10,
            )


if __name__ == "__main__":
    app()
