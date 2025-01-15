# Predictions API

This service provides an API for visualising predictions for candidate classifiers.

## Local Development

1. Make sure you have Docker and Docker Compose installed.

2. Build and run the service:

```bash
just run-predictions-api
```

The API service will start in a container and should be available at `http://localhost:8000`.

## Deployment

The API is deployed to AWS ECS Fargate. Here's the complete deployment process:

1. First, ensure you're using the correct AWS account:

```bash
# Configure Pulumi to use the labs AWS profile
pulumi config set aws:profile labs

# Verify you're using the correct AWS account
aws sts get-caller-identity --profile=labs
```

2. Deploy the infrastructure with Pulumi to create the necessary resources:

```bash
# Navigate to the infrastructure directory
cd predictions_api/infra

# Install dependencies
pip install -r requirements.txt

# Deploy and get the outputs
pulumi up --stack labs

# Export the values you'll need for the next steps
export ECR_REPOSITORY_URL=$(pulumi stack output ecr_repository_url --stack labs)
export AWS_REGION=$(pulumi stack output aws_region --stack labs)
export ECS_CLUSTER_NAME=$(pulumi stack output ecs_cluster_name --stack labs)
export ECS_SERVICE_NAME=$(pulumi stack output ecs_service_name --stack labs)
```

3. Build and push the Docker image to ECR using the values from step 1:

```bash
# Login to ECR
aws ecr get-login-password --region $AWS_REGION --profile labs | docker login --username AWS --password-stdin $ECR_REPOSITORY_URL

# Build the Docker image
docker build -t predictions-api .

# Tag the image for ECR
docker tag predictions-api:latest $ECR_REPOSITORY_URL:latest

# Push to ECR
docker push $ECR_REPOSITORY_URL:latest
```

4. Force a new deployment of the ECS service (if updating an existing deployment):

```bash
aws ecs update-service \
  --cluster $ECS_CLUSTER_NAME \
  --service $ECS_SERVICE_NAME \
  --force-new-deployment \
  --region $AWS_REGION \
  --profile labs
```

You can run `pulumi destroy --stack labs` to tear down the service when needed.

## Creating data for the API to consume and present

To create data for the API, you can run the `just predict Q123` with the relevant wikibase ID. This will create a set of predictions and classifiers in the `data/processed` directory.

To save the outputs to s3 for the deployed version of the API, you can run the same command with the `--save-to-s3` flag.
