.PHONY: install process-gst populate-wikibase knowledge-graph docs

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

process-gst:
	poetry run python scripts/process_gst.py

populate-wikibase:
	poetry run python scripts/populate_wikibase.py

knowledge-graph:
	{ \
	docker-compose up -d neo4j; \
	poetry run python scripts/populate_knowledge_graph.py; \
	}

docs:
	cd docs && poetry run mkdocs serve
