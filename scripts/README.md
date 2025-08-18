# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `uv run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.


## Updating a Classifier

First we run the training scripts. This will upload the classifier to s3 and link to it from its weights and biases project. You can then run the promote script which is used for promoting a model to primary within an aws environment. Promotion as adds the model to the weights and bias registry and setting it as primary gives it the environment alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
just train "Q123" --track --upload --aws-env sandbox
```

Then we promote:

```shell
just promote "Q123" --classifier_id abcd2345 --aws-env sandbox --primary
```

You can also achieve the above directly with:

```shell
just deploy-classifiers sandbox Q374
```

Or run a batch sequentially with:

```shell
just deploy-classifiers sandbox "Q374 Q473"
```

This is useful when you are already resolved that the trained model will become the new primary.


## Training Classifiers in Docker

This guide explains how to train classifiers using Docker containers with AWS integration. This may be desirable for developers as installing transformers (which is required for training our neural network based models) locally can be difficult; with system incompatibilities and version support issues being common.

## How It Works

The training process uses a locally built Docker image with volume mounts to maintain persistence and connectivity:

- **Local Image**: The Docker image is built locally using the `just build-image` command
- **Volume Mounts**: Key directories are mounted as volumes to persist changes and maintain access to external services
- **YAML Persistence**: Classifier specification updates are persisted back to the knowledge-graph repository
- **AWS Integration**: AWS CLI authentication is maintained through mounted credential volumes

## Prerequisites

- Docker installed and running
- AWS credentials configured locally
- Environment file (`.env`) with necessary configuration
- Access to the knowledge-graph repository

## Building the Docker Image

First, build the Docker image from the repository root:

```bash
just build-image
```

## Running the Training Container

### 1. Authenticate your CLI & Start the Container

This step will cache your token in `.aws/sso/cache`; this can be utilised by the aws cli when mounted into the container.

```bash
aws sso login --profile staging
```

Run the Docker container with the following command, mounting AWS credentials and classifier specs:

```bash
docker run \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  -v ~/.aws/sso/cache:/root/.aws/sso/cache:ro \
  -v $(pwd)/flows/classifier_specs/v2:/flows/classifier_specs/v2 \
  -e AWS_PROFILE=staging \
  -it kg-bumped-pytorch:latest /bin/sh
```

**Mount Points:**

- `~/.aws:/root/.aws:ro` - Read-only AWS credentials
- `~/.aws/sso/cache:/root/.aws/sso/cache:ro` - Read-only AWS SSO cache
- `$(pwd)/flows/classifier_specs/v2:/flows/classifier_specs/v2` - Classifier specifications

### 2. Install Dependencies

Once inside the container, install required system packages:

```bash
# Update package list and install unzip
apt-get update && apt-get install -y unzip
```

### 3. Install AWS CLI

Install the AWS CLI for S3 access:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws
```

### 4. Validate Installation

Verify AWS CLI is working correctly:

```bash
aws s3 ls
```

## Running the Training Script

Execute the training pipeline using the deploy script:

```bash
python scripts/deploy.py new \
  --aws-env prod \
  --get \
  --train \
  --promote \
  --wikibase-id Q1651
```

**Parameters:**

- `--aws-env staging` - The AWS environment to use
- `--get` - Fetch concept data from Wikibase
- `--train` - Train the classifier model
- `--promote` - Promote the trained model
- `--wikibase-id Q1651` - Specific concept ID to train


## Troubleshooting

- Ensure AWS credentials are properly configured locally prior to running the container
- Verify the `.env` file contains all required environment variables
- Check that the classifier specs directory is accessible from the container
- Confirm the target Wikibase ID exists and has associated training data
