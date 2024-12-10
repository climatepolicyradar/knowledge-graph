from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.config import processed_data_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

app = FastAPI(title="Predictions API")
base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=base_dir / "templates")

# Mount the static directory
static_dir = base_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

predictions_dir = processed_data_dir / "predictions"


def load_predictions(wikibase_id: WikibaseID, classifier: str) -> List[LabelledPassage]:
    """Read a JSONL file and return a list of predictions."""
    file_path = predictions_dir / wikibase_id / f"{classifier}.jsonl"
    with open(file_path, encoding="utf-8") as f:
        predictions = [LabelledPassage.model_validate_json(line) for line in f]
    return predictions


def get_available_classifiers(wikibase_id: WikibaseID) -> List[str]:
    """Get list of available classifiers for a concept."""
    concept_dir = predictions_dir / wikibase_id
    if not concept_dir.exists():
        return []
    return [f.stem for f in concept_dir.glob("*.jsonl")]


def get_available_regions(predictions: List[LabelledPassage]) -> List[str]:
    """Extract unique World Bank regions from predictions."""
    regions: set[str] = set()
    for prediction in predictions:
        region = prediction.metadata.get("world_bank_region")
        if region and isinstance(region, str) and region.strip():
            regions.add(region)
    return sorted(list(regions))


def get_available_translated_statuses(predictions: List[LabelledPassage]) -> List[str]:
    """Extract unique translated statuses from predictions."""
    translated_statuses: set[str] = set()
    for prediction in predictions:
        translated = prediction.metadata.get("translated")
        if translated and isinstance(translated, str) and translated.strip():
            translated_statuses.add(translated)
    return sorted(list(translated_statuses))


def get_available_corpora(predictions: List[LabelledPassage]) -> List[str]:
    """Extract unique corpora from predictions."""
    corpora: set[str] = set()
    for prediction in predictions:
        corpus = prediction.metadata.get("dataset_name")
        if corpus and isinstance(corpus, str) and corpus.strip():
            corpora.add(corpus)
    return sorted(list(corpora))


def get_available_concepts() -> List[dict]:
    """Get list of available concepts with their details."""
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
async def index(request: Request):
    """Display homepage with list of available concepts."""
    concepts = get_available_concepts()
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


@app.get("/{wikibase_id}/{classifier}")
async def get_predictions_html(
    request: Request, wikibase_id: WikibaseID, classifier: str
):
    """Return predictions for a concept and classifier in HTML format."""
    try:
        predictions = load_predictions(wikibase_id, classifier)
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(wikibase_id)
        regions = get_available_regions(predictions)
        translated_statuses = get_available_translated_statuses(predictions)
        available_corpora = get_available_corpora(predictions)

        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "predictions": predictions,
                "wikibase_id": wikibase_id,
                "classifier": classifier,
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
            detail=f"Predictions not found for concept {wikibase_id} and classifier {classifier}",
        )


@app.get("/{wikibase_id}/{classifier}/json")
async def get_predictions_json(wikibase_id: WikibaseID, classifier: str):
    """Return predictions for a concept and classifier in JSON format."""
    try:
        predictions = load_predictions(wikibase_id, classifier)
        return {
            "wikibase_id": wikibase_id,
            "classifier": classifier,
            "total_count": len(predictions),
            "predictions": predictions,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Predictions not found for concept {wikibase_id} and classifier {classifier}",
        )
