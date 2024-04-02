# knowledge-graph

Infrastructure, tools, and scripts for managing Climate Policy Radar's concepts store and knowledge graph.

The concepts store is a wikibase instance, used by the policy team to manage individual climate concepts and the relationships between them.

## Developing

- `make install` will set up a local environment for development using poetry and pre-commit.
- `make process-gst` will process the GST data and save a static representation of it in json format at `data/processed/concepts.json`. Assumes that the [global-stocktake](https://github.com/climatepolicyradar/global-stocktake) repo has been cloned in the same directory as this repo.
- `make populate-wikibase` will populate the wikibase instance with the processed GST data
