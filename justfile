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
    poetry run python scripts/build_dataset.py

# fetch metadata and labelled passages for a specific wikibase ID
get-concept id: install
    poetry run python scripts/get_concept.py {{id}}

# train a model for a specific wikibase ID
train id: (get-concept id)
    poetry run python scripts/train.py {{id}}

# evaluate a model for a specific wikibase ID
evaluate id: (get-concept id)
    poetry run python scripts/evaluate.py {{id}}

# run a model for a specific wikibase ID on a supplied string
label id string: (train id)
    poetry run python scripts/label.py {{id}} {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id: (train id) build-dataset
    poetry run python scripts/predict.py {{id}}

# sample a set of passages from the dataset for a specific wikibase ID
sample id n: (train id) (predict id)
    poetry run python scripts/sample.py {{id}} {{n}}

# push a sampled set of passages to argilla for a specific wikibase ID
push-to-argilla id n: (sample id n)
    poetry run python scripts/push_to_argilla.py {{id}}

# run the full pipeline for a specific wikibase ID
reissue-labelling-task id n: (sample id n)
    just get-concept {{id}}
    just train {{id}}
    just predict {{id}}
    just sample {{id}} {{n}}
    just push-to-argilla {{id}}
