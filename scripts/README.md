# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `uv run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.

## Updating a Classifier

First we run the training scripts. This will upload the classifier to s3 and link to it from its weights and biases project. You can then run the promote script which is used for promoting a model to primary within an aws environment. Promotion as adds the model to the weights and bias registry and setting it as primary gives it the environment alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
just train "Q123" --track --aws-env sandbox
```

> [!NOTE]
> Promoting requires at least once classifiers profile. Set it in the metadata.

Then we promote:

```shell
just promote "Q123" --classifier-id abcd2345 --aws-env sandbox --add-classifiers-profiles primary
```

Finally, to update the classifier specs:

```shell
just update-inference-classifiers --aws-envs sandbox
```

You can also achieve the above directly with:

```shell
just deploy-classifier Q374 sandbox
```

Or run a batch sequentially with:

```shell
just deploy-classifiers "Q374 Q473" sandbox
```

This is useful when you are already resolved that the trained model will become the new primary.

### Prevent a model from running on specific sources

You can add a source to the classifiers metadata with the following, this will prevent documents with the source from having inference run with this classifier:

```shell
just classifier-metadata Q123 abcd2345 sandbox --add-dont-run-on sabin
```

Add and override the current list:

```shell
just classifier-metadata Q123 abcd2345 sandbox --clear-dont-run-on --add-dont-run-on sabin --add-dont-run-on gef
```

Clear the list to allow the classifier to run on anything

```shell
just classifier-metadata Q123 abcd2345 sandbox --clear-dont-run-on
```

Add a requirement for the classifier to run in a compute environment that has a GPU (or use clear-require-gpu to remove and revert to using a cpu)

```shell
just classifier-metadata Q123 abcd2345 sandbox --add-require-gpu
```

For times when its necessary to update every promoted classifier that is mentioned in the spec for an environment, you can run the following:

```shell
just classifier-metadata-entire-env sandbox --add-dont-run-on sabin
```

At least one classififiers profile is required for promotion. You could set one like:

```shell
just classifier-metadata Q57 jq7535b6 sandbox --add-classifiers-profiles primary
```

If a classifier specification should no longer be used, the inverse of a promotion should be done—a demotion. This will demote the latest version of the AWS env specified in the registry by removing the tag.

```shell
just demote Q57 --aws-env sandbox
```

If you require a specific registry version to be demoted, you can add the registry version as a parameter (note: this version is not the same as the project model version:

```shell
just demote Q57 --wandb-registry-version v10 --aws-env sandbox
```

Then, update the classifier specifications as per usual.

## Training Classifiers in Docker

This guide explains how to train classifiers using Docker containers with AWS integration. This may be desirable for developers as installing transformers (which is required for training our neural network based models) locally can be difficult; with system incompatibilities and version support issues being common.

### How It Works

The training process uses a locally built Docker image with volume mounts to maintain persistence and connectivity:

- **Local Image**: The Docker image is built locally using the `just build-image` command
- **Volume Mounts**: Key directories are mounted as volumes to persist changes and maintain access to external services
- **YAML Persistence**: Classifier specification updates are persisted back to the knowledge-graph repository
- **AWS Integration**: AWS CLI authentication is maintained through mounted credential volumes

### Prerequisites

- Docker installed and running
- AWS credentials configured locally
- Environment file (`.env`) with necessary configuration
- Access to the knowledge-graph repository

### Building the Docker Image

First, build the Docker image from the repository root:

```bash
just build-image
```

### Running the Training Container

#### 1. Authenticate your CLI & Start the Container

This step will cache your token in `.aws/sso/cache`; this can be utilised by the aws cli when mounted into the container.

```bash
aws sso login --profile staging
```

#### 2. Run the docker container

Run the Docker container with the following command, mounting AWS credentials and classifier specs:

```bash
docker run \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  -v ~/.aws/sso/cache:/root/.aws/sso/cache:ro \
  -v $(pwd)/flows/classifier_specs/v2:/flows/classifier_specs/v2 \
  -e AWS_PROFILE=staging \
  -it ${DOCKER_REGISTRY}/${DOCKER_REPOSITORY}:${VERSION} /bin/sh
```

**Mount Points:**

- `~/.aws:/root/.aws:ro` - Read-only AWS credentials
- `~/.aws/sso/cache:/root/.aws/sso/cache:ro` - Read-only AWS SSO cache
- `$(pwd)/flows/classifier_specs/v2:/flows/classifier_specs/v2` - Classifier specifications

#### 3. Validate Installation + Authentication against AWS

Verify AWS CLI is working correctly:

```bash
aws s3 ls
```

## Running the Training Script

Execute the training pipeline using the deploy script:

```bash
uv run deploy new \
  --aws-env prod \
  --train \
  --promote \
  --wikibase-id Q1651
```

Note: If the classifier spec files in the local repo do not update after running the deploy script in docker then simply come out of the docker container and run `just update-inference-classifiers`.

## Troubleshooting

- Ensure AWS credentials are properly configured locally prior to running the container
- Verify the `.env` file contains all required environment variables
- Check that the classifier specs directory is accessible from the container
- Confirm the target Wikibase ID exists and has associated training data

### Weights and biases: Permission Error

If you experience the following error:

```shell
requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://api.wandb.ai/graphql

ERROR Permission denied to access team/classifier/version
```

To resolve:

- Ensure permissions for weights and biases are granted for the user for projects as well as for the  [model registry](https://docs.wandb.ai/guides/registry/configure_registry/)
- If access for only projects is granted then the `train` scripts will be successful since they are accessing the projects however will fail during `promote` when the registry is accessed
