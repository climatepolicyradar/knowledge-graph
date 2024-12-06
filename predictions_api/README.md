# Predictions API

A FastAPI application that uses Jinja2 templates to renderpredictions from candidate concept classifiers.

The data for each concept is stored in a static JSON file in `data/processed/predictions/{concept_id}.json`.

The API has two endpoints:

- `GET /predictions/{concept_id}`
- `GET /predictions/{concept_id}/json`

The first endpoint renders the predictions in HTML format, with a Jinja2 template.

The second endpoint returns the predictions in raw JSON format.
