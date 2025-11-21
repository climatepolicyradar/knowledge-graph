set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
aws_region := env_var_or_default('AWS_REGION', 'eu-west-1')
aws_profile := env_var_or_default('AWS_PROFILE', 'labs')
webapp_dir := 'vibe-checker/webapp'
infra_dir := 'vibe-checker/infra'

# Install webapp dependencies via npm
install-webapp:
    cd {{webapp_dir}} && npm ci

# Lint webapp code via ESLint
lint-webapp:
    cd {{webapp_dir}} && npm run lint

# Deploy AWS infrastructure with Pulumi
deploy-infra:
    cd {{infra_dir}} && pulumi up
    
# Build the Next.js webapp Docker image for Linux x86-64 (AWS)
build-webapp:
    cd {{webapp_dir}} && docker buildx build --platform linux/amd64 -t vibe-checker-webapp .

serve-webapp:
    cd {{webapp_dir}} && npm run dev

# Push the Docker image to ECR
push-webapp:
    cd {{infra_dir}} && export REPO_URL=$(pulumi stack output repository_url) && \
    cd ../{{webapp_dir}} && docker tag vibe-checker-webapp:latest $REPO_URL:latest && \
    aws ecr get-login-password --region {{aws_region}} --profile {{aws_profile}} | docker login --username AWS --password-stdin $REPO_URL && \
    docker push $REPO_URL:latest

# refresh the ecs task definition
refresh-webapp-task:
    cd {{infra_dir}} && aws ecs update-service --cluster $(pulumi stack output cluster_name) --service $(pulumi stack output service_name) --force-new-deployment --region {{aws_region}} --profile {{aws_profile}}

# Run all steps to deploy the webapp
deploy-webapp: build-webapp push-webapp refresh-webapp-task
