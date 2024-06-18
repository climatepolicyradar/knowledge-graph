# Prefect Flows

## Introduction

This directory contains prefect flows that can be developed in the knowledge graph repository. This allows one to seemlessy develop flows very close to the source code.

The prefect orchestration system is deployed in the cloud and thus for all instances of running flows / deployments you need to authenticate with the cloud. Until we all have personal accounts (from enterpise prefect) its recommended we use tokens for access. 

Login to prefect cloud, select the api key option and use the key provided in Bitwarden:

```shell
prefect cloud login 
```

### Run a flow locally, utilising local dependencies

The following example requires declaring a flow, then serving this locally as a deployment. You can then trigger the deployment via a command. The flow then runs locally in your environment.

Serve the deployment from local:
```shell
poetry run python -m flows.repo_info
```

Trigger the deployment:
```shell
prefect deployment run 'get-repo-info/my-first-deployment'
```

### Run a flow in the prefect worker in ECS

The following example runs a flow in the prefect worker that is deployed in an elastic container service cluster in the cloud.

Execute the flow:
```shell
poetry run python -m flows.repo_info
```

There are alternative interfaces for running flows. For example you can register your local terminal as a worker in a work pool and have that poll a work queue for jobs. If the above doesn't meet the requirements then we can look at expanding more. 


### Using KG Codebase

As proof that the prefect flows can utilise local dependencies and code please see the `flows/concepts.py` module.

```shell 
poetry run python -m flows.concepts
```
