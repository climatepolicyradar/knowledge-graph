set dotenv-load
export WANDB_DIR := "./data/wandb"
import "tests/local_vespa/local_vespa.just"

# Set the default command to list all available commands
default:
    @just --list

# install dependencies and set up the project
install:
    poetry install --with dev
    poetry run pre-commit install
    poetry run ipython kernel install --user

# install, but also include libraries for embeddings
install-for-embeddings: install
    poetry install --with embeddings

# test the project
test +OPTS="":
    poetry run pytest {{OPTS}}

# test the project, excluding tests that rely on a local vespa instance
test-without-vespa:
    poetry run pytest -m 'not vespa'

# update the snapshots for the tests
test-snapshot-update:
    poetry run pytest --snapshot-update

# run linters and code formatters on relevant files
lint:
    poetry run pre-commit run --show-diff-on-failure

# run linters and code formatters on all files
lint-all:
    poetry run pre-commit run --all-files --show-diff-on-failure

# build a dataset of passages for sampling
build-dataset:
    poetry run python scripts/build_dataset/01_download_corporate_disclosures.py
    poetry run python scripts/build_dataset/02_download_litigation.py
    poetry run python scripts/build_dataset/03_parse.py
    poetry run python scripts/build_dataset/04_translate.py
    poetry run python scripts/build_dataset/05_add_geography.py
    poetry run python scripts/build_dataset/06_merge.py
    poetry run python scripts/build_dataset/07_create_balanced_dataset_for_sampling.py

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

# demote a model for a specific wikibase ID
demote id aws_env:
    poetry run demote --wikibase-id {{id}} --aws-env {{aws_env}}

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

# generate a markdown report of classifier performance
generate-report wikibase-ids:
    poetry run python scripts/generate_report.py --wikibase-ids {{wikibase-ids}}

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

export-env-vars:
	export $(cat .env | xargs)

prefect-login: export-env-vars
	prefect cloud login -k ${PREFECT_API_KEY}

deploy: prefect-login
    just deploy-deployments
    just deploy-automations

deploy-deployments: prefect-login
	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}
	python -m deployments

deploy-automations: prefect-login
	python -m automations

# Run inference over documents in a pipeline bucket
infer +OPTS="":
    poetry run infer {{OPTS}}

# Run inference over documents in the sandbox pipeline bucket
infer-sandbox +OPTS="":
    just infer --aws_env sandbox {{OPTS}}

# Run inference over documents in the labs pipeline bucket
infer-labs +OPTS="":
    just infer --aws_env labs {{OPTS}}

# Update what classifiers we are going to run for during inference.
# Checks for latest versions of classifiers in wandb and updates spec files
update-inference-classifiers +OPTS="":
    poetry run update_inference_classifiers {{OPTS}}
