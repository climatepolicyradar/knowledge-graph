set dotenv-load

# install dependencies and set up the project
install:
    uv sync --extra dev
    uv run pre-commit install
    uv run ipython kernel install --user

# test the project
test:
    uv run pytest

# update the snapshots for the tests
test-snapshot-update:
    uv run pytest --snapshot-update

# run linters and code formatters
lint:
    uv run pre-commit run --all-files --show-diff-on-failure

# build a dataset of passages
build-dataset:
    uv run python scripts/build_dataset.py

# fetch metadata and labelled passages for a specific wikibase ID
get-concept id:
    uv run python scripts/get_concept.py --wikibase-id {{id}}

# train a model for a specific wikibase ID
train id +OPTS="":
    uv run scripts/train.py --wikibase-id {{id}} {{OPTS}}

# evaluate a model for a specific wikibase ID
evaluate id:
    uv run python scripts/evaluate.py --wikibase-id {{id}}

# promote a model for a specific wikibase ID
promote id +OPTS="":
    uv run scripts/promote.py --wikibase-id {{id}} {{OPTS}}

# run a model for a specific wikibase ID on a supplied string
label id string:
    uv run python scripts/label.py --wikibase-id {{id}} --input-string {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id:
    uv run python scripts/predict.py --wikibase-id {{id}}

# sample a set of passages from the dataset for a specific wikibase ID
sample id:
    uv run python scripts/sample.py --wikibase-id {{id}}

# push a sampled set of passages to argilla for a specific wikibase ID
push-to-argilla id usernames workspace:
    uv run python scripts/push_to_argilla.py --wikibase-id {{id}} --usernames {{usernames}} --workspace {{workspace}}

# run the full pipeline for a specific wikibase ID
create-labelling-task id usernames workspace:
    just get-concept --wikibase-id {{id}}
    just train --wikibase-id {{id}}
    just predict --wikibase-id {{id}}
    just sample --wikibase-id {{id}}
    just push-to-argilla --wikibase-id {{id}} --usernames {{usernames}} --workspace {{workspace}}

# visualise IAA, model vs gold-standard agreement, and positive predictions on the full dataset
visualise-labels id:
    uv run python scripts/visualise_labels.py --wikibase-id {{id}}

analyse-classifier id: (get-concept id) (train id) (predict id) (evaluate id) (visualise-labels id)
