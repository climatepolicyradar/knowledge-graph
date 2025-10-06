# Prefect Flows

## Introduction

This directory contains prefect flows for the knowledge graph repository. Utilising our orchestration layer defined in the orchestrator [repository](https://github.com/climatepolicyradar/orchestrator), this allows one to seamlessly develop flows very close to the source code in this repository whilst deploying to our cloud based infrastructure.

The prefect orchestration system is deployed in the cloud and thus for all instances of running flows / deployments you need to authenticate with the cloud. To authenticate with prefect cloud create an account with your work email and ask to be added to the organisation. Then follow the Prefect [documentation](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud#how-to-connect-to-prefect-cloud) here to authenticate via cli.


### Run a flow locally, utilising local dependencies

Follow this Prefect [How-To-Guide](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-a-flow) to run a flow locally. Note if you want your flow to access AWS Infrastructure like s3 objects; you can authenticate your cli with AWS cloud and your local flow will assume the permissions of your terminal session. 

### Run a flow in CPR's AWS Cloud Environment

We currently have deployed within the Production Prefect workspace [Elastic Container Service](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html) workpools, [coiled](https://coiled.io/) work pools and are currently integrating prefect managed work pools for all of our AWS environments (production, staging, sandbox, labs). To run a flow on any of these work pools simply configure the deployment with the work pool parameter. Deployments can be created in a number of ways as per the Prefect [documentation](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments). Our deployments are currently declared using the `.deploy` method of the flow object in the `deployments.py` module within this repository.
