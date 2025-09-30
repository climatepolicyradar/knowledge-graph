
# build a dataset of passages for sampling
build-dataset n="10000":
    poetry run python scripts/build_dataset.py --n {{n}}

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

# Deploy (Get, train & deploy) new model to primary for the given AWS environment.
# Example: just deploy-classifiers sandbox Q123
deploy-classifier id aws_env +OPTS="":
    uv run deploy new \
        {{id}} \
        --aws-env {{aws_env}} \
        --train \
        --promote \
        {{OPTS}}

# Deploy (Get, train & deploy) new models to primary for the given AWS environment.
# Example: just deploy-classifiers sandbox 'Q123 Q368 Q374 Q404 Q412'
deploy-classifiers ids aws_env +OPTS="":
    #!/bin/bash
    set -e

    # Convert the ids argument to a list of --wikibase-id arguments
    ids_args=""
    for id in {{ids}}; do
      ids_args="$ids_args --wikibase-id $id"
    done

    uv run deploy new \
        --aws-env {{aws_env}} \
        $ids_args \
        --train \
        --promote \
        {{OPTS}}

# run a model for a specific wikibase ID on a supplied string
label id string:
    uv run python scripts/label.py --wikibase-id {{id}} --input-string {{string}}

# find instances of the concept in a set of passages for a specific wikibase ID
predict id +OPTS="":
    uv run python scripts/predict.py --wikibase-id {{id}} {{OPTS}}

# run classifiers on balanced dataset for vibe checking (legacy predict functionality)
vibe-check-predict id +OPTS="":
    uv run python scripts/vibe_checker/predict.py --wikibase-id {{id}} {{OPTS}}

# sample a set of passages from the dataset for a specific wikibase ID
sample id:
    uv run python scripts/sample.py --wikibase-id {{id}}

# sample passages using multiple classifiers for a specific wikibase ID
sample-using-classifiers id +OPTS="":
    uv run python scripts/sample_using_classifiers.py --wikibase-id {{id}} {{OPTS}}

# extend an existing dataset in argilla with additional passages for a specific wikibase ID
extend-dataset id workspace:
    uv run python scripts/extend_an_existing_dataset_in_argilla.py --wikibase-id {{id}} --workspace-name {{workspace}}

# push a sampled set of passages to argilla for a specific wikibase ID
push-to-argilla id workspace:
    uv run python scripts/push_a_fresh_dataset_to_argilla.py --wikibase-id {{id}} --workspace-name {{workspace}}

# generate an HTML report of classifier performance
generate-report wikibase-ids:
    uv run python scripts/generate_report.py --wikibase-ids {{wikibase-ids}}

# visualise IAA, model vs gold-standard agreement, and positive predictions on the full dataset
visualise-labels id:
    uv run python scripts/visualise_labels.py --wikibase-id {{id}}

analyse-classifier id: (get-concept id) (train id) (predict id) (evaluate id)

# Run inference over documents in a pipeline bucket
infer +OPTS="":
    uv run infer {{OPTS}}

# Run inference over documents in the sandbox pipeline bucket
infer-sandbox +OPTS="":
    just infer --aws_env sandbox {{OPTS}}

# Run inference over documents in the labs pipeline bucket
infer-labs +OPTS="":
    just infer --aws_env labs {{OPTS}}


# Update the metadata for a classifier
classifier-metadata wikibase_id classifier_id aws_env +OPTS="":
    uv run classifier-metadata update \
        --wikibase-id {{wikibase_id}} \
        --classifier-id {{classifier_id}} \
        --aws-env {{aws_env}} \
        {{OPTS}}

# Update the same metadata for multiple classifiers in one aws env
classifier-metadata-entire-env aws_env +OPTS="":
    uv run classifier-metadata update-entire-env \
        --aws-env {{aws_env}} \
        {{OPTS}}

# Update what classifiers we are going to run for during inference.
# Checks for latest versions of classifiers in wandb and updates spec files
update-inference-classifiers +OPTS="":
    uv run update-inference-classifiers {{OPTS}}

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
