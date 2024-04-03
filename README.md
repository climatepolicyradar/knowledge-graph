# 🕸️ Knowledge graph

Infrastructure, tools, and scripts for managing Climate Policy Radar's concepts store and knowledge graph.

The concepts store is a wikibase instance, used by the policy team to manage individual climate concepts and the relationships between them.

The knowledge graph is a (local, for now) neo4j instance, used to visualise and query the concepts from the concepts store, along with additional nodes and metadata (eg. the relationships between concepts and the documents that mention them).

## Developing

- `make install` will set up a local environment for development using poetry and pre-commit.
- `make process-gst` will process the GST data and save a static representation of it in json format at `data/processed/concepts.json`. Assumes that the [global-stocktake](https://github.com/climatepolicyradar/global-stocktake) repo has been cloned in the same directory as this repo.
- `make populate-wikibase` will populate the wikibase instance with the processed GST data
- `make knowledge-graph` will start a neo4j instance, and pull the data from the wikibase instance into the neo4j instance. It can then be accessed at `http://localhost:7474/browser/` with the credentials `neo4j`/`neo4j`
