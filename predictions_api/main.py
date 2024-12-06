from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scripts.config import processed_data_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage

app = FastAPI(title="Predictions API")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

predictions_dir = processed_data_dir / "predictions"


def load_predictions(wikibase_id: WikibaseID) -> List[LabelledPassage]:
    """Read a JSONL file and return a list of predictions."""
    file_path = predictions_dir / f"{wikibase_id}.jsonl"
    with open(file_path, encoding="utf-8") as f:
        predictions = [LabelledPassage.model_validate_json(line) for line in f]
    return predictions


@app.get("/predictions/{wikibase_id}")
async def get_predictions_html(request: Request, wikibase_id: WikibaseID):
    """Return predictions for a concept in HTML format."""
    try:
        predictions = load_predictions(wikibase_id)
        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "predictions": predictions,
                "wikibase_id": wikibase_id,
                "total_count": len(predictions),
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Concept {wikibase_id} not found")


@app.get("/predictions/{wikibase_id}/json")
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
