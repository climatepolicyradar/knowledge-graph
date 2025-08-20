import io
import json
import shutil
from pathlib import Path

import boto3
import pandas as pd
import typer
import yaml
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from rich.console import Console

from scripts.config import processed_data_dir
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

S3_BUCKET_NAME = "cpr-knowledge-graph-vibe-check"
PERSISTENT_PREFIX = "persistent"

session = boto3.Session(region_name="eu-west-1")
s3 = session.client("s3")


def load_sample_dataset() -> pd.DataFrame:
    """Load the sample dataset from S3. Fails if dataset cannot be loaded."""
    try:
        with console.status("🚚 Loading the sample dataset from S3"):
            bytes_from_s3 = s3.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=PERSISTENT_PREFIX + "/passages_dataset.feather",
            )["Body"].read()
            sample_passages = pd.read_feather(io.BytesIO(bytes_from_s3))
        console.log(
            f"📚 Loaded {len(sample_passages)} passages from the sample dataset"
        )
        return sample_passages
    except (ClientError, NoCredentialsError) as e:
        console.log(f"❌ Failed to load sample dataset from S3: {e}", style="red")
        raise SystemExit(1) from e
    except Exception as e:
        console.log(f"❌ Unexpected error loading sample dataset: {e}", style="red")
        raise SystemExit(1) from e


def discover_persistent_concept_ids() -> set[WikibaseID]:
    """Discover all concept IDs that have persistent experimental results in S3."""
    concept_ids = set()
    try:
        with console.status(
            "🔍 Discovering concepts with persistent experimental results"
        ):
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=S3_BUCKET_NAME, Prefix=f"{PERSISTENT_PREFIX}/"
            )
            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Parse concept ID from path like "persistent/Q123/6npuynbx.json"
                    parts = key.split("/")
                    if len(parts) == 3 and parts[0] == PERSISTENT_PREFIX:
                        concept_id = parts[1]
                        if concept_id and WikibaseID._validate(concept_id):
                            concept_ids.add(concept_id)

            console.log(f"📁 Found persistent results for {len(concept_ids)} concepts")
    except (ClientError, NoCredentialsError) as e:
        console.log(f"⚠️  Failed to discover persistent concepts: {e}", style="yellow")
    except Exception as e:
        console.log(
            f"❌ Unexpected error discovering persistent concepts: {e}", style="red"
        )

    return concept_ids


def fetch_persistent_predictions(
    concept_id: WikibaseID,
) -> dict[str, list[LabelledPassage]]:
    """Fetch persistent experimental predictions for a concept from S3."""
    persistent_predictions = {}
    try:
        # list all JSON files in the persistent directory for this concept
        prefix = f"{PERSISTENT_PREFIX}/{concept_id}/"
        with console.status(f"🔍 Checking for persistent predictions for {concept_id}"):
            response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

            if "Contents" not in response:
                console.log(f"📭 No persistent predictions found for {concept_id}")
                return persistent_predictions

            json_files = [
                obj["Key"]
                for obj in response["Contents"]
                if obj["Key"].endswith(".json")
            ]

            if not json_files:
                console.log(
                    f"📭 No JSON files found in persistent storage for {concept_id}"
                )
                return persistent_predictions

            console.log(
                f"📁 Found {len(json_files)} persistent prediction files for {concept_id}"
            )

            for key in json_files:
                classifier_id = Path(key).stem
                try:
                    with console.status(f"📥 Loading predictions from {key}"):
                        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                        content = obj["Body"].read().decode("utf-8")

                        predictions: list[LabelledPassage] = [
                            LabelledPassage.model_validate(json.loads(line))
                            for line in content.strip().split("\n")
                            if line.strip()
                        ]

                        if predictions:
                            persistent_predictions[classifier_id] = predictions
                            console.log(
                                f"✅ Loaded {len(predictions)} persistent predictions for {classifier_id}"
                            )
                        else:
                            console.log(
                                f"⚠️  No valid predictions found in {key}",
                                style="yellow",
                            )

                except (ClientError, json.JSONDecodeError, ValidationError) as e:
                    console.log(
                        f"❌ Failed to load persistent predictions from {key}: {e}",
                        style="red",
                    )
                    continue

    except (ClientError, NoCredentialsError) as e:
        console.log(
            f"⚠️  Failed to access S3 for persistent predictions: {e}", style="yellow"
        )
    except Exception as e:
        console.log(
            f"❌ Unexpected error fetching persistent predictions: {e}", style="red"
        )

    return persistent_predictions


def generate_predictions_for_classifier(
    classifier: Classifier, sample_passages: pd.DataFrame
) -> list[LabelledPassage]:
    """Generate predictions for a single classifier, with error handling."""
    predictions = []

    try:
        with console.status(f"🔮 Generating predictions for {classifier}"):
            for _, row in sample_passages.iterrows():
                try:
                    text = row.get("text_block.text", "") or ""
                    if not text.strip():
                        continue

                    if predicted_spans := classifier.predict(text):
                        predictions.append(
                            LabelledPassage(
                                text=text,
                                spans=predicted_spans,
                                metadata=row.to_dict(),
                            )
                        )
                except Exception as e:
                    console.log(
                        f"⚠️  Error processing passage for {classifier}: {e}",
                        style="yellow",
                    )
                    continue

        console.log(f"🔍 Generated {len(predictions)} predictions for {classifier}")

    except Exception as e:
        console.log(
            f"❌ Failed to generate predictions for {classifier}: {e}", style="red"
        )

    return predictions


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


def save_predictions_json(
    concept_dir: Path, classifier_id: str, predictions: list[LabelledPassage]
) -> None:
    """Save predictions to JSON file with error handling."""
    try:
        json_content = "\n".join([p.model_dump_json() for p in predictions])
        json_path = concept_dir / f"{classifier_id}.json"
        json_path.write_text(json_content, encoding="utf-8")
    except Exception as e:
        console.log(f"❌ Failed to save JSON for {classifier_id}: {e}", style="red")


def generate_concept_html(
    concept_dir: Path,
    concept: Concept,
    all_predictions: dict[str, list[LabelledPassage]],
) -> None:
    """Generate HTML files for a concept with error handling."""
    try:
        # Generate concept index page
        concept_template = templates.get_template("concept.html")
        classifiers_info = {
            classifier_id: f"Classifier {classifier_id}"
            for classifier_id in all_predictions.keys()
        }

        concept_html = concept_template.render(
            wikibase_id=concept.wikibase_id,
            concept_str=str(concept),
            wikibase_url=concept.wikibase_url,
            classifiers=classifiers_info,
        )

        (concept_dir / "index.html").write_text(concept_html, encoding="utf-8")

        # Generate predictions pages for each classifier
        predictions_template = templates.get_template("predictions.html")

        for classifier_id, predictions in all_predictions.items():
            try:
                # Gather filter options
                regions = get_available_regions(predictions)
                translated_statuses = get_available_translated_statuses(predictions)
                corpora = get_available_corpora(predictions)

                predictions_html = predictions_template.render(
                    predictions=predictions,
                    wikibase_id=concept.wikibase_id,
                    wikibase_url=concept.wikibase_url,
                    classifier_id=classifier_id,
                    classifier_name=classifiers_info.get(classifier_id, classifier_id),
                    concept_str=str(concept),
                    regions=regions,
                    translated_statuses=translated_statuses,
                    corpora=corpora,
                    classifiers=classifiers_info,
                    total_count=len(predictions),
                )

                html_path = concept_dir / f"{classifier_id}.html"
                html_path.write_text(predictions_html, encoding="utf-8")

            except Exception as e:
                console.log(
                    f"❌ Failed to generate HTML for {classifier_id}: {e}", style="red"
                )
                continue

    except Exception as e:
        console.log(
            f"❌ Failed to generate HTML for concept {concept.wikibase_id}: {e}",
            style="red",
        )


@app.command()
def main():
    """Generate the vibe check static site with persistent experimental results support."""

    # Copy static assets
    static_src = base_dir / "static"
    static_dest = dist_dir / "static"
    if static_src.exists():
        shutil.copytree(static_src, static_dest)

    # Load sample dataset
    sample_passages = load_sample_dataset()

    concept_ids_from_config = set()
    try:
        with open(base_dir / "config.yml", "r") as f:
            classifier_specs = yaml.safe_load(f)
            concept_ids_from_config = set(
                [WikibaseID(concept_id) for concept_id in classifier_specs]
            )
        console.log(f"📋 Found {len(concept_ids_from_config)} wikibase IDs in config")
    except Exception as e:
        console.log(
            f"⚠️  Failed to load classifier specs: {e}. Continuing with persistent results only.",
            style="yellow",
        )

    # Discover concept IDs with persistent experimental results
    persistent_concept_ids = discover_persistent_concept_ids()

    # Combine all concept IDs (production + persistent experimental)
    all_concept_ids = concept_ids_from_config.union(persistent_concept_ids)
    console.log(f"🎯 Total unique concepts to process: {len(all_concept_ids)}")

    if not all_concept_ids:
        console.log(
            "❌ No concepts found in either production specs or persistent storage",
            style="red",
        )
        return

    # Load concepts from Wikibase (with graceful handling for missing concepts)
    try:
        wikibase = WikibaseSession()

        # Try to load all concepts, but handle failures gracefully
        concepts: dict[WikibaseID, Concept] = {}
        valid_concept_ids = []

        for concept_id in all_concept_ids:
            try:
                if concept := wikibase.get_concept(
                    wikibase_id=concept_id,
                    include_recursive_has_subconcept=True,
                    include_labels_from_subconcepts=True,
                ):
                    concepts[concept_id] = concept
                    valid_concept_ids.append(concept_id)
                else:
                    console.log(
                        f"⚠️  Concept {concept_id} not found in Wikibase", style="yellow"
                    )
            except Exception as e:
                console.log(
                    f"⚠️  Failed to load concept {concept_id}: {e}", style="yellow"
                )
                continue

        console.log(f"✅ Successfully loaded {len(concepts)} concepts from Wikibase")

    except Exception as e:
        console.log(f"❌ Failed to connect to Wikibase: {e}", style="red")
        return

    # Filter out targets concepts
    concepts = {
        concept_id: concept
        for concept_id, concept in concepts.items()
        if concept_id not in ["Q1651", "Q1652", "Q1653"]
    }

    console.log(
        f"🎯 Processing {len(concepts)} concepts ({len(concept_ids_from_config & set(concepts.keys()))} production + {len(persistent_concept_ids & set(concepts.keys()))} persistent-only)"
    )

    # Phase 1: Generate all JSON predictions
    console.log("📊 Phase 1: Generating all predictions data", style="bold blue")

    all_concept_predictions: dict[WikibaseID, dict[str, list[LabelledPassage]]] = {}

    for concept_id, concept in concepts.items():
        console.log(f"🔄 Processing concept {concept_id}: {concept}")
        concept_predictions = {}

        # Generate predictions from default classifier (if this concept is in production)
        if concept_id in concept_ids_from_config:
            try:
                classifier = ClassifierFactory.create(concept)
                console.log(f"🤖 Created classifier: {classifier}")

                if predictions := generate_predictions_for_classifier(
                    classifier, sample_passages
                ):
                    concept_predictions[classifier.id] = predictions

            except Exception as e:
                console.log(
                    f"❌ Failed to create/run classifier for {concept_id}: {e}",
                    style="red",
                )
        else:
            console.log(
                f"📝 Concept {concept_id} has no production classifier, checking persistent results only"
            )

        # Fetch persistent experimental predictions
        persistent_predictions = fetch_persistent_predictions(concept_id)
        concept_predictions.update(persistent_predictions)

        if concept_predictions:
            all_concept_predictions[concept_id] = concept_predictions
            console.log(
                f"✅ Total classifiers for {concept_id}: {len(concept_predictions)}"
            )
        else:
            console.log(f"⚠️  No predictions available for {concept_id}", style="yellow")

    # Phase 2: Generate all HTML and structure for /dist
    console.log(
        "🌐 Phase 2: Generating HTML files and site structure", style="bold blue"
    )

    # Generate index page
    try:
        index_template = templates.get_template("index.html")
        valid_concepts = [
            concept
            for concept_id, concept in concepts.items()
            if concept_id in all_concept_predictions
        ]
        index_html = index_template.render(
            concepts=valid_concepts, total_count=len(valid_concepts)
        )
        (dist_dir / "index.html").write_text(index_html, encoding="utf-8")
        console.log(f"📄 Generated index page with {len(valid_concepts)} concepts")
    except Exception as e:
        console.log(f"❌ Failed to generate index page: {e}", style="red")

    # Generate concept pages and save JSON files
    for concept_id, concept_predictions in all_concept_predictions.items():
        concept = concepts[concept_id]
        concept_dir = dist_dir / str(concept_id)
        concept_dir.mkdir(exist_ok=True)

        # Save all JSON files
        for classifier_id, predictions in concept_predictions.items():
            save_predictions_json(concept_dir, classifier_id, predictions)

        # Generate HTML files
        generate_concept_html(concept_dir, concept, concept_predictions)

        console.log(
            f'📄 Generated pages for "{concept}" with {len(concept_predictions)} classifiers'
        )

    # Save the sample dataset to the dist directory for reference
    try:
        sample_passages.to_feather(dist_dir / "passages_dataset.feather")
        console.log("💾 Saved sample dataset to dist directory")
    except Exception as e:
        console.log(f"⚠️  Failed to save sample dataset: {e}", style="yellow")

    console.log(f"✅ Successfully generated static site in {dist_dir}", style="green")
    console.log(
        "🚀 You can now run the site locally using `just serve-static-site vibe_check`",
        style="green",
    )


if __name__ == "__main__":
    typer.run(main)
