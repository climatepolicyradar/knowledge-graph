# Predictions API

A FastAPI application that uses Jinja2 templates to render predictions from candidate concept classifiers.

The data for each concept is stored in a static JSON file in `data/processed/predictions/{concept_id}.json`.

The API has three main endpoints:

- `GET /` renders a list of all available concepts in HTML using a Jinja2 template.
- `GET /predictions/{concept_id}` renders the predictions for a concept in HTML using a Jinja2 template.
- `GET /predictions/{concept_id}/json` returns the predictions in raw JSON format.

## Running the API

From the root of the repository, run:

```bash
poetry install --with predictions_api
```

```bash
poetry run uvicorn predictions_api.main:app --reload
```
