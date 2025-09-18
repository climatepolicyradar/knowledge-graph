import random
from typing import Optional

from rich.console import Console
from rich.table import Table

from knowledge_graph.classifier.large_language_model import LLMClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.ensemble import Ensemble
from knowledge_graph.ensemble.metrics import Disagreement
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.wikibase import WikibaseSession

console = Console()


def load_concept_and_labelled_passages(
    concept_id: WikibaseID,
    wikibase: WikibaseSession,
    argilla: ArgillaSession,
    max_passages_per_concept: Optional[int] = None,
) -> Concept:
    """Load a concept and its labelled passages."""

    concept = wikibase.get_concept(concept_id)
    concept.labelled_passages = argilla.pull_labelled_passages(concept)

    if max_passages_per_concept is not None:
        concept.labelled_passages = concept.labelled_passages[:max_passages_per_concept]

    console.log(
        f"🧠 Loaded concept [bold white]{concept}[/bold white] with {len(concept.labelled_passages)} labelled passages."
    )

    return concept


def main(concept_id: str):
    """Prototype active learning script."""

    random.seed(100)

    console.log(f"Loading concept {concept_id}")
    wikibase = WikibaseSession()
    argilla = ArgillaSession()

    concept = load_concept_and_labelled_passages(
        concept_id=WikibaseID(concept_id),
        wikibase=wikibase,
        argilla=argilla,
    )

    console.log("Getting text...")
    # TODO: this should be text sampled from elsewhere – not the concept's labelled
    # passages!
    texts: list[str] = [passage.text for passage in concept.labelled_passages]
    texts = texts[:10]

    console.log("Creating ensemble of LLM classifiers")
    base_llm_classifier = LLMClassifier(concept, model_name="gpt-4.1-mini")

    # TODO: it seems like get_variants might be nicer here as we have control over
    # the properties of the group
    classifiers_for_ensemble = [base_llm_classifier] + [
        base_llm_classifier.get_variant() for _ in range(9)
    ]
    ensemble = Ensemble(concept, classifiers=classifiers_for_ensemble)

    uncertainty_metric = Disagreement()
    uncertainty_metric_str = f"Uncertainty ({uncertainty_metric.name})"
    console.log(f"Calculating uncertainty using metric {uncertainty_metric.name}")

    spans_per_text_per_classifier = ensemble.predict_batch(texts)

    uncertainty_per_text = [
        uncertainty_metric(spans) for spans in spans_per_text_per_classifier
    ]

    console.log(
        f"Rendering table of texts ordered by decreasing {uncertainty_metric_str}"
    )
    rows = list(zip(texts, uncertainty_per_text))
    rows.sort(key=lambda item: float(item[1]), reverse=True)

    table = Table(title="Active Learning: Texts by Uncertainty", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column(
        uncertainty_metric_str, justify="right", style="magenta", no_wrap=True
    )
    table.add_column("Text", justify="left", style="white")

    for idx, (text, score) in enumerate(rows, start=1):
        table.add_row(str(idx), f"{float(score):.3f}", text)

    console.print(table)

    console.log(
        "I'll now wave my magic wand and send the ones I'm less sure about to a human.\nThe ones I'm really sure about we can send straight to a BERT model ✨"
    )


if __name__ == "__main__":
    main("Q760")
