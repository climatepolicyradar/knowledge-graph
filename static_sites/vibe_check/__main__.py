import io
import shutil
from pathlib import Path

import boto3
import pandas as pd
import typer
import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from scripts.config import processed_data_dir, root_dir
from src.classifier import Classifier, ClassifierFactory
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

load_dotenv()
app = typer.Typer()

console = Console(stderr=True)

# Set up Jinja2 templates
base_dir = Path(__file__).parent
templates = Environment(loader=FileSystemLoader(base_dir / "templates"))

# Set up the input and output directories
predictions_dir = processed_data_dir / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)
dist_dir = base_dir / "dist"
if dist_dir.exists():
    shutil.rmtree(dist_dir)
dist_dir.mkdir(parents=True)

# Load sample passages from S3
s3 = boto3.client("s3", region_name="eu-west-1")
with console.status("🚚 Loading the sample dataset"):
    bytes_from_s3 = s3.get_object(
        Bucket="cpr-knowledge-graph-vibe-check", Key="passages_dataset.feather"
    )["Body"].read()
    sample_passages = pd.read_feather(io.BytesIO(bytes_from_s3))
console.log(f"📚 Loaded {len(sample_passages)} passages from the sample dataset")


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

    with open(root_dir / "flows" / "classifier_specs" / "prod.yaml", "r") as f:
        classifier_specs = yaml.safe_load(f)

    concept_ids = list(
        set([classifier_spec.split(":")[0] for classifier_spec in classifier_specs])
    )
    concepts: dict[WikibaseID, Concept] = dict(
        zip(concept_ids, wikibase.get_concepts(wikibase_ids=concept_ids))
    )

    # ignore targets concepts
    concepts = {
        WikibaseID(concept_id): concept
        for concept_id, concept in concepts.items()
        if concept_id
        not in [
            "Q1651",
            "Q1652",
            "Q1653",
        ]
    }
    classifiers: dict[WikibaseID, Classifier] = {}
    for concept_id, concept in concepts.items():
        classifier = ClassifierFactory.create(concept)
        classifiers[concept_id] = classifier
        console.log(f"🤖 Created a {classifier} for {concept_id}")

    # Generate index page
    index_template = templates.get_template("index.html")
    index_html = index_template.render(
        concepts=list(concepts.values()), total_count=len(concepts)
    )
    (dist_dir / "index.html").write_text(index_html)
    console.log(f"📄 Generated index page with {len(concepts)} concepts")

    # Generate concept pages
    for i, (concept_id, classifier) in enumerate(classifiers.items()):
        console.log(f"🤔 Processing concept {i + 1}/{len(classifiers)}: {classifier}")
        concept = concepts[concept_id]
        concept_template = templates.get_template("concept.html")
        concept_html = concept_template.render(
            wikibase_id=concept.wikibase_id,
            concept_str=str(concept),
            wikibase_url=concept.wikibase_url,
            classifiers={classifier.id: classifier.name},
        )

        concept_dir = dist_dir / str(concept.wikibase_id)
        concept_dir.mkdir(exist_ok=True)
        (concept_dir / "index.html").write_text(concept_html)
        console.log(f'📄 Generated concept page for "{concept}"')

        # Generate predictions using the classifier
        predictions: list[LabelledPassage] = []
        for _, row in sample_passages.iterrows():
            text = row["text_block.text"] or ""
            predicted_spans = classifier.predict(text)
            if predicted_spans:
                predictions.append(
                    LabelledPassage(
                        text=text,
                        spans=predicted_spans,
                        metadata=row.to_dict(),
                    )
                )
        console.log(
            f'🔍 Found {len(predictions)} instances of "{concept}" in the sample dataset'
        )

        # Generate the predictions page HTML
        predictions_template = templates.get_template("predictions.html")
        predictions_html = predictions_template.render(
            predictions=predictions,
            wikibase_id=concept.wikibase_id,
            wikibase_url=concept.wikibase_url,
            classifier_id=classifier.id,
            classifier_name=classifier.name,
            concept_str=str(concept),
            regions=get_available_regions(predictions),
            translated_statuses=get_available_translated_statuses(predictions),
            corpora=get_available_corpora(predictions),
            classifiers={classifier.id: classifier.name},
            total_count=len(predictions),
        )

        # Write the HTML version
        (concept_dir / f"{classifier.id}.html").write_text(predictions_html)

        # Write the JSONL version
        json_content = "\n".join([p.model_dump_json() for p in predictions])
        (concept_dir / f"{classifier.id}.json").write_text(json_content)

        console.log(f'📄 Generated predictions page for "{concept}"')

    console.log(f"✅ Successfully generated static site in {dist_dir}", style="green")
    console.log(
        "🚀 You can now run the site locally using `just serve-static-site vibe_check`",
        style="green",
    )


if __name__ == "__main__":
    typer.run(main)
