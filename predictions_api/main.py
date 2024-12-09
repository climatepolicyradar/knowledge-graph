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


def load_predictions(wikibase_id: WikibaseID) -> List[LabelledPassage]:
    """Read a JSONL file and return a list of predictions."""
    file_path = predictions_dir / f"{wikibase_id}.jsonl"
    with open(file_path, encoding="utf-8") as f:
        predictions = [LabelledPassage.model_validate_json(line) for line in f]
    return predictions


def get_available_concepts() -> List[dict]:
    """Get list of available concepts with their details."""
    concepts = []
    wikibase = WikibaseSession()

    for file_path in predictions_dir.glob("*.jsonl"):
        wikibase_id = WikibaseID(file_path.stem)
        try:
            concept = wikibase.get_concept(wikibase_id)
            concepts.append(
                {
                    "id": wikibase_id,
                    "label": concept.preferred_label,
                    "str": str(concept),
                    "description": concept.description,
                    "url": concept.wikibase_url,
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
async def get_predictions_html(request: Request, wikibase_id: WikibaseID):
    """Return predictions for a concept in HTML format."""
    try:
        predictions = load_predictions(wikibase_id)
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(wikibase_id)
        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "predictions": predictions,
                "wikibase_id": wikibase_id,
                "total_count": len(predictions),
                "wikibase_url": concept.wikibase_url,
                "concept_str": str(concept),
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Concept {wikibase_id} not found")


@app.get("/{wikibase_id}/json")
async def get_predictions_json(wikibase_id: WikibaseID):
    """Return predictions for a concept in JSON format."""
    try:
        predictions = load_predictions(wikibase_id)
        return {
            "wikibase_id": wikibase_id,
            "total_count": len(predictions),
            "predictions": predictions,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Concept {wikibase_id} not found")
