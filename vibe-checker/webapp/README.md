# Vibe Checker Webapp

A Next.js-based web application for testing and evaluating classifiers against real-world policy documents.

## Local Development

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

The app will be available at `http://localhost:3000`.

The app depends on being able to fetch data from an S3 bucket, hosted in the `labs` account and managed by the code in the [infra](../infra) directory. You'll need to authenticate with AWS and set the `BUCKET_NAME` environment variable:

```bash
aws sso login --profile labs
export BUCKET_NAME=$(cd ../infra && pulumi stack output bucket_name --stack labs)
```

## Deployment

The webapp is deployed to AWS ECS Fargate. The deployment is managed by Pulumi IaC in the [infra](../infra) directory.

Assuming the infrastructure is already deployed, there are three major steps in the deployment of the webapp: building the Docker image, pushing it to ECR, and refreshing the ECS task with the new image. Run these from the `vibe-checker/` directory:

```bash
just build-webapp
just push-webapp
just refresh-webapp-task
```

Alternatively, you can run all three in one step:

```bash
just deploy-webapp
```

When the deployment is complete, you can monitor the deployment logs by running:

```bash
cd infra
aws logs tail /ecs/vibe-checker --follow --profile labs --region eu-west-1
```

If the app is running happily, you should see logs like this:

```text
2025-11-11T17:38:46.880000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e > webapp@0.1.0 start
2025-11-11T17:38:46.880000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e > next start
2025-11-11T17:38:48.677000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e    ▲ Next.js 15.5.2
2025-11-11T17:38:48.678000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e    - Local:        http://localhost:3000
2025-11-11T17:38:48.678000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e    - Network:      http://169.254.172.2:3000
2025-11-11T17:38:48.678000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e  ✓ Starting...
2025-11-11T17:38:49.778000+00:00 ecs/vibe-checker/320c2c3e42174603b624e0a08fd5e11e  ✓ Ready in 1917ms
```
