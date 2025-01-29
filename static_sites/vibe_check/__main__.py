import shutil
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

app = typer.Typer()

console = Console(stderr=True)

# Set up Jinja2 templates
base_dir = Path(__file__).parent
templates = Environment(loader=FileSystemLoader(base_dir / "templates"))

# Set up the input and output directories
predictions_dir = processed_data_dir / "predictions"
dist_dir = base_dir / "dist"
if dist_dir.exists():
    shutil.rmtree(dist_dir)
dist_dir.mkdir(parents=True)


def load_predictions(
    wikibase_id: WikibaseID, classifier_id: str
) -> list[LabelledPassage]:
    """Read a JSONL file and return a list of predictions."""
    file_path = predictions_dir / wikibase_id / f"{classifier_id}.jsonl"
    if not file_path.exists():
        console.log(
            f"Warning: No predictions file found for {wikibase_id}/{classifier_id}",
            style="yellow",
        )
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            predictions = [LabelledPassage.model_validate_json(line) for line in f]
        return predictions
    except Exception as e:
        console.log(
            f"Error loading predictions for {wikibase_id}/{classifier_id}: {str(e)}",
            style="yellow",
        )
        return []


def get_available_classifiers(wikibase_id: WikibaseID) -> dict[str, str]:
    """Get list of available classifiers for a concept."""
    concept_classifier_dir = classifier_dir / wikibase_id
    if not concept_classifier_dir.exists():
        return {}

    classifiers = {}
    for path in concept_classifier_dir.glob("*.pickle"):
        classifier_id = path.stem
        classifier_name = str(Classifier.load(path))
        classifiers[classifier_id] = classifier_name

    return classifiers


def get_available_regions(predictions: list[LabelledPassage]) -> list[str]:
    """Extract unique World Bank regions from predictions."""
    regions: set[str] = set()
    for prediction in predictions:
        region = prediction.metadata.get("world_bank_region")
        if region and isinstance(region, str) and region.strip():
            regions.add(region)
    return sorted(list(regions))


def get_available_translated_statuses(predictions: list[LabelledPassage]) -> list[str]:
    """Extract unique translated statuses from predictions."""
    translated_statuses: set[str] = set()
    for prediction in predictions:
        # Convert boolean or string value to consistent string format
        is_translated = prediction.metadata.get("translated", False)
        if isinstance(is_translated, str):
            # Handle string values like "True" or "False"
            translated_statuses.add(
                "True" if is_translated.lower() == "true" else "False"
            )
        else:
            # Handle boolean values
            translated_statuses.add(str(bool(is_translated)))
    return sorted(list(translated_statuses))


def get_available_corpora(predictions: list[LabelledPassage]) -> list[str]:
    """Extract unique corpora from predictions."""
    corpora: set[str] = set()
    for prediction in predictions:
        corpus = prediction.metadata.get("document_metadata.corpus_type_name")
        if corpus and isinstance(corpus, str) and corpus.strip():
            corpora.add(corpus)
    return sorted(list(corpora))


@app.command()
def main():
    # Copy static assets
    static_src = base_dir / "static"
    static_dest = dist_dir / "static"
    if static_src.exists():
        shutil.copytree(static_src, static_dest)

    # Get all concepts with classifiers
    wikibase = WikibaseSession()
    concepts = []

    # Only look for concepts that have predictions
    for concept_dir in predictions_dir.iterdir():
        if not concept_dir.is_dir():
            continue

        wikibase_id = WikibaseID(concept_dir.name)
        try:
            concept = wikibase.get_concept(wikibase_id)
            classifiers = get_available_classifiers(wikibase_id)
            concepts.append(
                {
                    "id": wikibase_id,
                    "label": concept.preferred_label,
                    "str": str(concept),
                    "description": concept.description,
                    "url": concept.wikibase_url,
                    "classifiers": classifiers,
                    "n_classifiers": len(classifiers),
                }
            )
            console.log(
                f'Found concept "{concept}" with {len(classifiers)} classifiers'
            )

        except Exception as e:
            console.log(
                f"Skipping concept {wikibase_id} due to error: {str(e)}",
                style="yellow",
            )
            continue

    # Generate index page
    index_template = templates.get_template("index.html")
    index_html = index_template.render(concepts=concepts, total_count=len(concepts))
    (dist_dir / "index.html").write_text(index_html)
    console.log(f"Generated index page with {len(concepts)} concepts")

    # Generate concept pages
    for concept_data in concepts:
        wikibase_id = concept_data["id"]
        concept_template = templates.get_template("concept.html")
        concept_html = concept_template.render(
            wikibase_id=wikibase_id,
            concept_str=concept_data["str"],
            wikibase_url=concept_data["url"],
            classifiers=concept_data["classifiers"],
        )

        concept_dir = dist_dir / str(wikibase_id)
        concept_dir.mkdir(exist_ok=True)
        (concept_dir / "index.html").write_text(concept_html)
        console.log(f"Generated concept page for {wikibase_id}")

        # Generate predictions pages for each classifier
        for classifier_id in concept_data["classifiers"]:
            try:
                predictions = load_predictions(wikibase_id, classifier_id)
                predictions_template = templates.get_template("predictions.html")
                predictions_html = predictions_template.render(
                    predictions=predictions,
                    wikibase_id=wikibase_id,
                    wikibase_url=concept_data["url"],
                    classifier_id=classifier_id,
                    classifier_name=concept_data["classifiers"][classifier_id],
                    concept_str=concept_data["str"],
                    regions=get_available_regions(predictions),
                    translated_statuses=get_available_translated_statuses(predictions),
                    corpora=get_available_corpora(predictions),
                    classifiers=concept_data["classifiers"],
                )

                # Write the HTML version
                (concept_dir / f"{classifier_id}.html").write_text(predictions_html)

                # Write the JSONL version
                json_content = "\n".join([p.model_dump_json() for p in predictions])
                (concept_dir / f"{classifier_id}.json").write_text(json_content)

                console.log(
                    f"Generated predictions page for {wikibase_id}/{classifier_id} (HTML and JSONL)"
                )
            except Exception as e:
                console.log(
                    f"Failed to generate predictions for {wikibase_id}/{classifier_id}: {str(e)}",
                    style="yellow",
                )

    console.log(f"Successfully generated static site in {dist_dir}", style="green")
    console.log(
        "You can now run the site locally using `just serve-static-site vibe_check`",
        style="green",
    )


if __name__ == "__main__":
    typer.run(main)
