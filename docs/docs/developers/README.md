# Developers

## Setup

- `make install` will set up a local environment for development using poetry and pre-commit.
- `make process-gst` will process the GST data and save a static representation of it in json format at `data/processed/concepts.json`. Assumes that the [global-stocktake](https://github.com/climatepolicyradar/global-stocktake) repo has been cloned in the same directory as this repo.
- `make populate-wikibase` will populate the wikibase instance with the processed GST data
- `make knowledge-graph` will start a neo4j instance, and pull the data from the wikibase instance into the neo4j instance. It can then be accessed at `http://localhost:7474/browser/` with the credentials `neo4j`/`neo4j`
