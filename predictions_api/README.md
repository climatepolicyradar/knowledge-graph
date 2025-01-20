# Predictions API

This service provides an API for visualising predictions for candidate classifiers. The service is a FastAPI app which serves jinja2 HTML templates as well as json.

## Local Development

### Via Uvicorn (probably better/faster if you want to make changes to the app code)

Make sure you have a local development environment set up:

```bash
just install --with predictions_api
```

Then start the API with:

```bash
poetry run uvicorn predictions_api.app.main:app --host 0.0.0.0 --port 80 --reload 
```

It should be available at `http://localhost`.

### With Docker (will reproduce the container used in the deployed version of the API)

1. Make sure you have Docker and Docker Compose installed.

2. Build and run the service:

```bash
docker compose up --build predictions_api
```

The API service will start in a container and should be available at `http://localhost`.

## Without data from s3

By default, the API will sync the data from an s3 bucket on startup (see [Creating data for the API](#creating-data-for-the-api)).

If you want to run the API with data which is already locally available (ie different to what's in s3), you can set the `SKIP_S3_SYNC` environment variable to `true` in your local environment, or in your `docker-compose.yml` file. This will skip the sync step and use the data which is already in your `data/processed` directory.

## Deploying the service

The API is deployed to AWS ECS Fargate. Here's the complete deployment process:

### 1. Ensure you're using the correct AWS account

```bash
# Configure Pulumi to use the labs AWS profile
pulumi config set aws:profile labs

# Verify you're using the correct AWS account
aws sts get-caller-identity --profile=labs
```

### 2. Deploy the infrastructure with Pulumi to create the necessary resources

```bash
# Navigate to the infrastructure directory
cd predictions_api/infra

# Install dependencies
pip install -r requirements.txt

# Deploy and get the outputs
pulumi up --stack labs

# Export the values you'll need for the next steps
export ECR_REPOSITORY_URL="$(pulumi stack output ecr_repository_url --stack labs)"
export AWS_REGION="$(pulumi stack output aws_region --stack labs)"
export ECS_CLUSTER_NAME="$(pulumi stack output ecs_cluster_name --stack labs)"
export ECS_SERVICE_NAME="$(pulumi stack output ecs_service_name --stack labs)"
```

### 3. Build and push the Docker image to ECR

```bash
# Login to ECR
aws ecr get-login-password --region "$AWS_REGION" --profile labs | docker login --username AWS --password-stdin "$ECR_REPOSITORY_URL"

# Build the Docker image (from root of the repo)
docker build -t predictions-api -f predictions_api/Dockerfile .

# Tag the image for ECR
docker tag "predictions-api:latest" "${ECR_REPOSITORY_URL}:latest"

# Push to ECR
docker push "${ECR_REPOSITORY_URL}:latest"
```

### 4. Force a new deployment of the ECS service

Only necessary if updating an existing deployment.

```bash
aws ecs update-service \
  --cluster "$ECS_CLUSTER_NAME" \
  --service "$ECS_SERVICE_NAME" \
  --force-new-deployment \
  --region "$AWS_REGION" \
  --profile labs
```

### 5. Get the service URL (this may take a minute after deployment)

```bash
# Get the task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster "$ECS_CLUSTER_NAME" \
  --service-name "$ECS_SERVICE_NAME" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'taskArns[0]' \
  --output text)

# Get the ENI ID
ENI_ID=$(aws ecs describe-tasks \
  --cluster "$ECS_CLUSTER_NAME" \
  --tasks "$TASK_ARN" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text)

# Get the public DNS name
PUBLIC_DNS=$(aws ec2 describe-network-interfaces \
  --network-interface-ids "$ENI_ID" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'NetworkInterfaces[0].Association.PublicDnsName' \
  --output text)

echo "Service is available at: http://$PUBLIC_DNS"
```

## Tearing down the service

Navigate to the `infra` directory and run `pulumi destroy --stack labs` to tear down the service.

NB this won't delete the S3 bucket where predictions/models are stored, so you'll need to do that manually.

## Updating the service

If you make code changes which you want to deploy to the service, you'll need to follow the steps in [Deploying the service](#deploying-the-service) to build and push a new Docker image to ECR, and then force a new deployment of the ECS service.

## Adding new data

To create data for the API, you can run the `just predict Q123` with the relevant wikibase ID. This will create a set of predictions and classifiers in the `data/processed` directory, sufficient for a local development version of the API.

To save the outputs to s3 for the deployed version of the API, you can run the same command with the `--save-to-s3` flag.

After pushing new data to S3, you'll need to restart the API service so it can sync the latest data. Since you're not changing the application code, you only need to force a new deployment of the existing ECS task - follow [Step 4](#4-force-a-new-deployment-of-the-ecs-service) in the [deployment instructions](#deploying-the-service).
