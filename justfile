set dotenv-load

# install dependencies and set up the project
install:
    poetry install
    poetry run pre-commit install
    poetry run ipython kernel install --user

# test the project's core classes
test:
    poetry run pytest

# run linters and code formatters
lint:
    poetry run pre-commit run --all-files

# build a dataset of passages
build-dataset:
    poetry run python scripts/atomic/build_dataset.py

# fetch metadata and labelled passages for a specific wikibase ID
get-concept id:
    poetry run python scripts/atomic/get_concept.py --wikibase-id {{id}}

# train a model for a specific wikibase ID
train id:
    poetry run python scripts/atomic/train.py --wikibase-id {{id}}

# evaluate a model for a specific wikibase ID
evaluate id:
    poetry run python scripts/atomic/evaluate.py --wikibase-id {{id}}

# run a model for a specific wikibase ID on a supplied string
label id string: (train id)
    poetry run python scripts/atomic/label.py --wikibase-id {{id}} {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id: (train id) build-dataset
    poetry run python scripts/atomic/predict.py --wikibase-id {{id}}

# sample a set of passages from the dataset for a specific wikibase ID
sample id: (train id) (predict id)
    poetry run python scripts/atomic/sample.py {{id}}

# push a sampled set of passages to argilla for a specific wikibase ID
push-to-argilla id: (sample id)
    poetry run python scripts/atomic/push_to_argilla.py {{id}}

# run the full pipeline for a specific wikibase ID
create-labelling-task id: (sample id)
    just get-concept {{id}}
    just train {{id}}
    just predict {{id}}
    just sample {{id}}
    just push-to-argilla {{id}}
