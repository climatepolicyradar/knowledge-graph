set dotenv-load
export WANDB_DIR := "./data/wandb"
import "tests/local_vespa/local_vespa.just"
import "tests/local_wikibase/local_wikibase.just"

# Set the default command to list all available commands
default:
    @just --list

# install dependencies and set up the project
install +OPTS="":
    uv sync --locked --extra dev --extra transformers --extra coiled {{OPTS}}
    uv run pre-commit install
    uv run ipython kernel install --user

# test the project
test +OPTS="":
    uv run pytest --disable-pytest-warnings --color=yes {{OPTS}}

# test the project, excluding tests that rely on a local Vespa instance
test-without-vespa +OPTS="":
    uv run pytest --disable-pytest-warnings --color=yes {{OPTS}} -m 'not vespa'

# test the project, excluding slow tests
test-without-slow +OPTS="":
    uv run pytest --disable-pytest-warnings --color=yes {{OPTS}} -m 'not slow'

# update the snapshots for the tests
test-snapshot-update +OPTS="":
    uv run pytest --snapshot-update {{OPTS}}

# run linters and code formatters on relevant files
lint:
    uv run pre-commit run --show-diff-on-failure

# run linters and code formatters on all files
lint-all:
    uv run pre-commit run --all-files --show-diff-on-failure

# build a dataset of passages for sampling
build-dataset:
    uv run python scripts/build_dataset/01_download_corporate_disclosures.py
    uv run python scripts/build_dataset/02_download_litigation.py
    uv run python scripts/build_dataset/03_parse.py
    uv run python scripts/build_dataset/04_translate.py
    uv run python scripts/build_dataset/05_add_geography.py
    uv run python scripts/build_dataset/06_merge.py
    uv run python scripts/build_dataset/07_create_balanced_dataset_for_sampling.py

# fetch metadata and labelled passages for a specific wikibase ID
get-concept id:
    uv run python scripts/get_concept.py --wikibase-id {{id}}

# train a model for a specific wikibase ID
train id +OPTS="":
    uv run train --wikibase-id {{id}} {{OPTS}}

# evaluate a model for a specific wikibase ID
evaluate id +OPTS="":
    uv run evaluate --wikibase-id {{id}} {{OPTS}}

# promote a model for a specific wikibase ID
promote id +OPTS="":
    uv run promote --wikibase-id {{id}} {{OPTS}}

# demote a model for a specific wikibase ID
demote id aws_env:
    uv run demote --wikibase-id {{id}} --aws-env {{aws_env}}

# run a model for a specific wikibase ID on a supplied string
label id string:
    uv run python scripts/label.py --wikibase-id {{id}} --input-string {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id +OPTS="":
    uv run python scripts/predict.py --wikibase-id {{id}} {{OPTS}}

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

# generate an HTML report of classifier performance
generate-report wikibase-ids:
    uv run python scripts/generate_report.py --wikibase-ids {{wikibase-ids}}

# visualise IAA, model vs gold-standard agreement, and positive predictions on the full dataset
visualise-labels id:
    uv run python scripts/visualise_labels.py --wikibase-id {{id}}

analyse-classifier id: (get-concept id) (train id) (predict id) (evaluate id)

build-image:
    docker build --progress=plain -t ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} .

run-image cmd="sh":
    docker run --rm -it ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} {{cmd}}

ecr-login:
  aws ecr --profile prod get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}

push-image:
    docker push ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION}

get-version:
    uv run python -c "import importlib.metadata; print(importlib.metadata.version('knowledge-graph'))"

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
    uv run infer {{OPTS}}

# Run inference over documents in the sandbox pipeline bucket
infer-sandbox +OPTS="":
    just infer --aws_env sandbox {{OPTS}}

# Run inference over documents in the labs pipeline bucket
infer-labs +OPTS="":
    just infer --aws_env labs {{OPTS}}

# Update what classifiers we are going to run for during inference.
# Checks for latest versions of classifiers in wandb and updates spec files
update-inference-classifiers +OPTS="":
    uv run update-inference-classifiers {{OPTS}}

# Run a static site locally
serve-static-site tool:
    uv run python -m http.server -d static_sites/{{tool}}/dist 8080

# Generate a static site
generate-static-site tool:
    uv run python -m static_sites.{{tool}}

# Deploy (Get, train & deploy) new models to primary for the given AWS environment.
# Example: just deploy-classifiers sandbox 'Q123 Q368 Q374 Q404 Q412'
deploy-classifiers aws_env ids:
    #!/bin/bash
    set -e

    # Convert the ids argument to a list of --wikibase-id arguments
    ids_args=""
    for id in {{ids}}; do
      ids_args="$ids_args --wikibase-id $id"
    done

    uv run python scripts/deploy.py new \
        --aws-env {{aws_env}} \
        $ids_args \
        --get \
        --train \
        --promote

# Does inference have results for accepted classifiers?
# Set STAGING_CACHE_BUCKET, or pass as argument:
audit-inference-staging bucket_name="${STAGING_CACHE_BUCKET:-}":
    uv run python scripts/audit/do_classifier_specs_have_results.py \
        staging {{bucket_name}}

# Does inference have results for accepted classifiers?
# Set PROD_CACHE_BUCKET, or pass as argument:
audit-inference-prod bucket_name="${PROD_CACHE_BUCKET:-}":
    uv run python scripts/audit/do_classifier_specs_have_results.py \
        prod {{bucket_name}}

# Do concepts for a doc align across sources - staging
# Set STAGING_CACHE_BUCKET in `.env`, or pass as argument:
# `just audit-doc-staging CCLW.executive.10491.5392 2025-06-03T16:35-eta4-esgaroth-ring`
audit-doc-staging document_id aggregator_run_identifier="latest" bucket_name="${STAGING_CACHE_BUCKET:-}":
    uv run python scripts/audit/do_outputs_align_for_a_document.py \
        {{document_id}} staging {{bucket_name}} \
        --aggregator-run-identifier {{aggregator_run_identifier}}

# Do concepts for a doc align across sources - prod
# Set PROD_CACHE_BUCKET in `.env`, or pass as argument:
# `just audit-doc-prod CCLW.document.i00001242.n0000 2025-06-03T16:35-eta4-esgaroth-ring`
audit-doc-prod document_id aggregator_run_identifier="latest" bucket_name="${PROD_CACHE_BUCKET:-}":
    uv run python scripts/audit/do_outputs_align_for_a_document.py \
        {{document_id}} prod {{bucket_name}} \
        --aggregator-run-identifier {{aggregator_run_identifier}}


# Check if passages in S3 align with Vespa
audit-s3-vespa-alignment +OPTS="":
    uv run python -m scripts.audit.do_s3_passages_align_with_vespa {{OPTS}}
