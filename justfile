set dotenv-load

# Set the default command to list all available commands
default:
    @just --list

# install dependencies and set up the project
install:
    poetry install --with dev
    poetry run pre-commit install
    poetry run ipython kernel install --user

# install, but also include libraries for embeddings
install-for-embeddings:
    just install
    poetry install --with embeddings

# test the project
test:
    poetry run pytest

# update the snapshots for the tests
test-snapshot-update:
    poetry run pytest --snapshot-update

# run linters and code formatters
lint:
    poetry run pre-commit run --all-files --show-diff-on-failure

# build a dataset of passages
build-dataset:
    poetry run python scripts/build_dataset.py

# fetch metadata and labelled passages for a specific wikibase ID
get-concept id:
    poetry run python scripts/get_concept.py --wikibase-id {{id}}

# train a model for a specific wikibase ID
train id +OPTS="":
    poetry run train --wikibase-id {{id}} {{OPTS}}

# evaluate a model for a specific wikibase ID
evaluate id +OPTS="":
    poetry run evaluate --wikibase-id {{id}} {{OPTS}}

# promote a model for a specific wikibase ID
promote id +OPTS="":
    poetry run promote --wikibase-id {{id}} {{OPTS}}

# run a model for a specific wikibase ID on a supplied string
label id string:
    poetry run python scripts/label.py --wikibase-id {{id}} --input-string {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id:
    poetry run python scripts/predict.py --wikibase-id {{id}}

# sample a set of passages from the dataset for a specific wikibase ID
sample id:
    poetry run python scripts/sample.py --wikibase-id {{id}}

# push a sampled set of passages to argilla for a specific wikibase ID
push-to-argilla id usernames workspace:
    poetry run python scripts/push_to_argilla.py --wikibase-id {{id}} --usernames {{usernames}} --workspace {{workspace}}

# run the full pipeline for a specific wikibase ID
create-labelling-task id usernames workspace:
    just get-concept --wikibase-id {{id}}
    just train --wikibase-id {{id}}
    just predict --wikibase-id {{id}}
    just sample --wikibase-id {{id}}
    just push-to-argilla --wikibase-id {{id}} --usernames {{usernames}} --workspace {{workspace}}

# visualise IAA, model vs gold-standard agreement, and positive predictions on the full dataset
visualise-labels id:
    poetry run python scripts/visualise_labels.py --wikibase-id {{id}}

analyse-classifier id: (get-concept id) (train id) (predict id) (evaluate id) (visualise-labels id)

build-image:
    docker build --progress=plain -t ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} .

ecr-login:
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}

push-image:
    docker push ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION}

get-version:
    poetry run python -c "import importlib.metadata; print(importlib.metadata.version('knowledge-graph'))"

export_env_vars:
	export $(cat .env | xargs)

prefect_login: export_env_vars
	prefect cloud login -k ${PREFECT_API_KEY}

deploy: prefect_login
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}
	python -m deployments

# Run inference over documents in a pipeline bucket
infer +OPTS="":
    poetry run python scripts/infer.py {{OPTS}}

# Run inference over documents in the sandbox pipeline bucket
infer-sandbox +OPTS="":
    just infer --aws_env sandbox {{OPTS}}

# Run inference over documents in the labs pipeline bucket
infer-labs +OPTS="":
    just infer --aws_env labs {{OPTS}}
