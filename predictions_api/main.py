import logging
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mypy_boto3_s3.client import S3Client

from scripts.config import aws_region, classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

logger = logging.getLogger("uvicorn")


def sync_s3_to_local() -> None:
    """Sync the prediction-visualisation S3 bucket to the local file system"""
    bucket_name = "prediction-visualisation"
    session = boto3.Session(profile_name="labs")
    s3_client: S3Client = session.client("s3", region_name=aws_region)

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting sync from s3://{bucket_name} to {processed_data_dir}")

    try:
        # Check whether the bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
    except s3_client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            raise ValueError(f"Bucket {bucket_name} does not exist") from e
        elif error_code == "403":
            raise ValueError(
                f"Missing permission to access bucket {bucket_name}"
            ) from e
        else:
            raise ValueError(
                f"Error accessing bucket {bucket_name}: {error_code}"
            ) from e

    files_synced = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            local_path = processed_data_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket_name, key, str(local_path))
            files_synced += 1

    logger.info(f"Successfully synced {files_synced} files from s3://{bucket_name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    try:
        sync_s3_to_local()
    except ValueError as e:
        logger.warning(f"S3 sync failed - {str(e)}")
    yield


app = FastAPI(title="Predictions API", lifespan=lifespan)
base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=base_dir / "templates")

# Mount the static directory
static_dir = base_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
predictions_dir = processed_data_dir / "predictions"


def load_predictions(
    wikibase_id: WikibaseID, classifier_id: str
) -> list[LabelledPassage]:
    """Read a JSONL file and return a list of predictions."""
    file_path = predictions_dir / wikibase_id / f"{classifier_id}.jsonl"
    with open(file_path, encoding="utf-8") as f:
        predictions = [LabelledPassage.model_validate_json(line) for line in f]
    return predictions


def get_available_classifiers(
    wikibase_id: WikibaseID,
) -> dict[str, str]:
    """Get list of available classifiers for a concept."""
    logger.info(f"Getting available classifiers for {wikibase_id}")
    concept_classifier_dir = classifier_dir / wikibase_id
    logger.info(f"Concept classifier dir: {concept_classifier_dir}")
    if not concept_classifier_dir.exists():
        return {}

    classifiers = {}
    for path in concept_classifier_dir.glob("*.pickle"):
        classifier_id = path.stem
        classifier_name = str(Classifier.load(path))
        classifiers[classifier_id] = classifier_name

    logger.info(f"Classifiers: {classifiers}")
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
        translated = prediction.metadata.get("translated")
        if translated and isinstance(translated, str) and translated.strip():
            translated_statuses.add(translated)
    return sorted(list(translated_statuses))


def get_available_corpora(predictions: list[LabelledPassage]) -> list[str]:
    """Extract unique corpora from predictions."""
    corpora: set[str] = set()
    for prediction in predictions:
        corpus = prediction.metadata.get("document_metadata.corpus_type_name")
        if corpus and isinstance(corpus, str) and corpus.strip():
            corpora.add(corpus)
    return sorted(list(corpora))


def get_available_concepts_with_classifiers() -> list[dict]:
    """Get list of available concepts with a list of classifiers and some metadata."""
    concepts = []
    wikibase = WikibaseSession()

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
                }
            )
        except Exception:
            continue

    return sorted(concepts, key=lambda x: x["id"])


@app.get("/")
async def get_index_page(request: Request):
    """Display homepage with list of available concepts."""
    concepts = get_available_concepts_with_classifiers()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "concepts": concepts,
            "total_count": len(concepts),
        },
    )


@app.get("/{wikibase_id}")
async def get_concept_page(request: Request, wikibase_id: WikibaseID):
    """Display concept page with available classifiers."""
    try:
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(wikibase_id)
        classifiers = get_available_classifiers(wikibase_id)

        if not classifiers:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for concept {wikibase_id}",
            )

        return templates.TemplateResponse(
            "concept.html",
            {
                "request": request,
                "wikibase_id": wikibase_id,
                "concept_str": str(concept),
                "wikibase_url": concept.wikibase_url,
                "classifiers": classifiers,
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Concept {wikibase_id} not found")


@app.get("/{wikibase_id}/{classifier_id}")
async def get_predictions_page(
    request: Request, wikibase_id: WikibaseID, classifier_id: str
):
    """Return predictions for a concept and classifier in HTML format."""
    try:
        predictions = load_predictions(wikibase_id, classifier_id)
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(wikibase_id)
        regions = get_available_regions(predictions)
        translated_statuses = get_available_translated_statuses(predictions)
        available_corpora = get_available_corpora(predictions)
        classifiers = get_available_classifiers(wikibase_id)

        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "predictions": predictions,
                "wikibase_id": wikibase_id,
                "classifier_id": classifier_id,
                "classifier_name": classifiers[classifier_id],
                "total_count": len(predictions),
                "wikibase_url": concept.wikibase_url,
                "concept_str": str(concept),
                "available_regions": regions,
                "available_translated_statuses": translated_statuses,
                "available_corpora": available_corpora,
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Predictions not found for concept {wikibase_id} and classifier {classifier_id}",
        )


@app.get("/{wikibase_id}/{classifier_id}/json")
async def get_predictions_json(wikibase_id: WikibaseID, classifier_id: str):
    """Return predictions for a concept and classifier in JSON format."""
    try:
        predictions = load_predictions(wikibase_id, classifier_id)
        return {
            "wikibase_id": wikibase_id,
            "classifier_id": classifier_id,
            "total_count": len(predictions),
            "predictions": predictions,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Predictions not found for concept {wikibase_id} and classifier {classifier_id}",
        )
