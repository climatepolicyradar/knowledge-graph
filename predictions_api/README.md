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
export ECR_REPOSITORY_URL="$(pulumi stack output ecr_repository_url --stack labs)"
export AWS_REGION="$(pulumi stack output aws_region --stack labs)"
export ECS_CLUSTER_NAME="$(pulumi stack output ecs_cluster_name --stack labs)"
export ECS_SERVICE_NAME="$(pulumi stack output ecs_service_name --stack labs)"
```

3. Build and push the Docker image to ECR using the values from step 1:

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

4. Force a new deployment of the ECS service (if updating an existing deployment):

```bash
aws ecs update-service \
  --cluster "$ECS_CLUSTER_NAME" \
  --service "$ECS_SERVICE_NAME" \
  --force-new-deployment \
  --region "$AWS_REGION" \
  --profile labs
```

5. Get the service URL (this may take a minute after deployment):

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

echo "Service is available at: http://$PUBLIC_DNS:8000"
```

You can run `pulumi destroy --stack labs` to tear down the service when needed. Note that if the ECR repository contains images, you'll need to delete them first:

### Getting the running container URL

```bash
# Get the task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster "$ECS_CLUSTER_NAME" \
  --service-name "$ECS_SERVICE_NAME" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'taskArns[0]' \
  --output text)

# Get the container instance ID
CONTAINER_INSTANCE_ID=$(aws ecs describe-tasks \
  --cluster "$ECS_CLUSTER_NAME" \
  --tasks "$TASK_ARN" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'tasks[0].containerInstanceArn' \
  --output text)

# Get the public IP address
PUBLIC_IP=$(aws ecs describe-container-instances \
  --cluster "$ECS_CLUSTER_NAME" \
  --container-instances "$CONTAINER_INSTANCE_ID" \
  --profile labs \
  --region "$AWS_REGION" \
  --query 'containerInstances[0].ec2InstanceId' \
  --output text)

echo "Container is available at: http://$PUBLIC_IP:8000"
```

## Creating data for the API to consume and present

To create data for the API, you can run the `just predict Q123` with the relevant wikibase ID. This will create a set of predictions and classifiers in the `data/processed` directory.

To save the outputs to s3 for the deployed version of the API, you can run the same command with the `--save-to-s3` flag.

You'll need to run the command to force a new deployment of the API service after this (see step 4 above).
