# Prefect Flows

## Introduction

This directory contains Prefect flows for the Knowledge Graph repository. By leveraging our orchestration layer maintained in a private [orchestrator repository](https://github.com/climatepolicyradar/orchestrator) within the Climate Policy Radar GitHub organization you can easily develop flows directly alongside the source code in this repository, while also deploying them seamlessly to our cloud infrastructure.

The Prefect orchestration system is deployed in the cloud, so you must authenticate with the cloud to run any flows or deployments. To gain access, you'll need a Prefect account and must be added to the Climate Policy Radar organisation. Once you have access, follow the Prefect [documentation](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud#how-to-connect-to-prefect-cloud) to authenticate via the CLI.


### Run a flow locally, utilising local dependencies

Follow this Prefect [How-To-Guide](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-a-flow) to run a flow locally. Note if you want your flow to access AWS Infrastructure like s3 objects; you can authenticate your cli with AWS cloud and your local flow will assume the permissions of your terminal session. 

_Below we see how to run a deployment locally using a flow within this repository._

Serve the deployment from local:

```shell
uv run python -m flows.repo_info
```

Trigger the deployment:

```shell
prefect deployment run 'get-repo-info/my-first-deployment'
```

### Run a flow in CPR's AWS Cloud Environment

Deployments can be created in a number of ways as per the Prefect [documentation](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments). Our deployments are currently declared using the `.deploy` method of the flow object in the `deployments.py` module within this repository. This should be used to define deployments to be run within the AWS Cloud environments. Deployment happens via CI/CD upon merge to main.

## Core + Flow convention

An "operation" (build a dataset, run a classifier, train a model, …) is split across two
layers:

- **`knowledge_graph/operations/*` — the core.** Reusable, _Prefect-free_ domain logic.
  This is the single source of truth for _what_ the operation does (querying,
  transforming, predicting). It ships in the `knowledge_graph` library wheel and can be
  imported anywhere — flows, notebooks, experiments — without pulling in Prefect.
- **`flows/*` — the orchestration.** A thin `@flow` that resolves credentials/config,
  calls the core function, and persists/logs results. This is _where and how_ the
  operation runs in the cloud.
- **`deployments.py` — the registry.** Registers flows as deployments (schedules,
  work pools, resources).

This replaced the older pattern where a `flows/X.py` held logic + flow + CLI together.
When migrating a remaining `scripts/` operation, follow this:

1. **Core logic goes in `knowledge_graph/operations/`.** Pure functions, no Prefect, no
   S3/orchestration glue. It returns data and lets the caller decide what to do with it
   (e.g. `run_build_dataset` returns dataframes; the flow uploads them, a local helper
   writes them to disk).

2. **The flow is a thin wrapper.** It resolves environment/credentials, calls the core
   function, and writes the output. Keep the flow small — `what` lives in the core.

3. **Respect the dependency direction** (never the reverse):

   ```text
   deployments.py -> flows/*
   flows/*        -> knowledge_graph/operations/*
   operations/*   -> knowledge_graph domain/library code
   ```

   Never `flows/* -> scripts/*`, never `operations/* -> flows/*`. If a helper needs a
   `flows.*` type (`Config`, `S3Uri`, `ClassifierSpec`, …) it belongs in `flows/`, not
   in operations.

4. **Credentials are environment-detected inside the flow.** Key-pair / SSM credentials
   when deployed; local config (AWS profiles, `~/.snowflake/config.toml`) when run
   locally.

5. **A CLI is optional — not the default architecture.** A full Typer CLI is only worth it
   where users genuinely need a rich local command (e.g. the inference deployment trigger);
   wire those into `pyproject.toml` `[project.scripts]`. For ad-hoc local runs, prefer a
   thin `just` recipe in `scripts/scripts.just` that calls the operation function via
   `python -c` (see `build-dataset`, `predict`, `predict-documents`), or call the operation
   directly from a notebook. A small local helper in the operation module (e.g.
   `build_dataset_locally`) is a good home for any logic the recipe would otherwise inline.

6. **Not everything is a flow.** A flow implies orchestration: scheduling, retries, a
   deployment, remote execution. Pure local dev utilities (e.g. `calculate_iaa`,
   `visualise_labels`) stay plain scripts or move into `knowledge_graph/` as library
   functions — don't wrap them in `@flow` for the sake of it.

### Current modules

| Operation | Core (`knowledge_graph/operations/`) | Flow (`flows/`) | Local run (`just`) |
| --- | --- | --- | --- |
| build_dataset | `run_build_dataset`, `build_dataset_locally` | `build_dataset_flow` → S3 (SSM creds) | `just build-dataset` / `build-dataset-corpus` → `build_dataset_locally`, writes to `data/processed/` |
| predict | `run_prediction`, `load_passages_from_snowflake`, `deduplicate_labelled_passages` | `predict_adhoc` (deployed), `predict_document_passages` (deployed, on-demand) | `just predict` / `predict-documents` → calls the operation directly (no Prefect) |
| infer | — not yet extracted; pure helpers (`document_passages`, `is_noop_document`, `_validate_spans`, …) still live in `flows/inference.py` | `inference` (+ batch variants) | `infer` — **deployment trigger only** (no local mode) |
| train | `run_training`, `train_classifier` (+ W&B/S3 helpers) | `train` (deployed; `train-on-cpu` / `train-on-gpu` variants) | `just train` → `scripts/train.py` CLI wraps `run_training` and dispatches the remote deployment for `--compute remote-cpu/remote-gpu` |
| sample | `run_sampling`, `CORPUS_TYPES` | `sample` → S3 (SSM creds) | `just sample` → `scripts/sample.py` CLI wraps `run_sampling`, loading the dataset from `data/processed/` |

### Known follow-ups

- `inference.py` has not been split yet — its per-document pure helpers
  (`document_passages`, `is_noop_document`, `_validate_spans`, …) still live in
  `flows/inference.py` alongside the orchestration. Extracting the pure helpers into
  `operations/infer.py` (with a `tests/operations/test_infer.py`) is the follow-up; helpers
  that need `flows.*` types should stay in `flows/`.
- The `analyse-classifier` just recipe has no prediction step: `just predict` now needs a
  classifier + passages artifact path, which the recipe chain (which only has an `id`)
  can't supply. Run `just predict` / `just predict-documents` separately for that step.
