# Prefect Flows

## Introduction

This directory contains prefect flows for the knowledge graph repository. Utilising our orchestration layer defined in a private  orchestrator [repository](https://github.com/climatepolicyradar/orchestrator) within the climatepolicyradar Github Organisation, this allows one to seamlessly develop flows very close to the source code in this repository whilst deploying to our cloud based infrastructure.

The prefect orchestration system is deployed in the cloud and thus for all instances of running flows / deployments you need to authenticate with the cloud. Access requires a prefect account and to be added to the Climate Policy Radar organisation. Then follow the Prefect [documentation](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud#how-to-connect-to-prefect-cloud) here to authenticate via cli.


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
