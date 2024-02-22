# knowledge-graph

Infrastructure and scripts for developing/managing Climate Policy Radar's knowledge graph.

The knowledge graph is a wikibase, used by the policy team to manage individual climate concepts and the relationships between them.

## Developing

- `make install` will set up a local environment for development using poetry and pre-commit.
- `make start-ec2` will start an EC2 instance for wikibase development.
- `make stop-ec2` will stop the EC2 instance.
- `make download-extensions` will download the necessary extensions for the wikibase deployment to `wikibase/extensions`.


## extra extension stuff

- need to download and unzip this one manually because it's not in the official list 
  - https://github.com/ProfessionalWiki/WikibaseEdtf/archive/refs/heads/master.zip
  - https://github.com/ProfessionalWiki/WikibaseLocalMedia/archive/refs/heads/master.zip
