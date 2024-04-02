.PHONY: install process-gst populate-wikibase

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

process-gst:
	poetry run python scripts/process_gst.py

populate-wikibase:
	poetry run python scripts/populate_wikibase.py
