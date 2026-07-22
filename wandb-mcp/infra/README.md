# Infrastructure for the W&B MCP server

Pulumi stack for deploying the Weights & Biases MCP server to AWS.

This section describes how to deploy the MCP server to AWS using Pulumi and AWS Fargate.

## Deploying

### Prerequisites

- Project dependencies installed by running `just install` from the project root.
- AWS account and [credentials configured for Pulumi](https://www.pulumi.com/docs/clouds/aws/get-started/begin/).
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running locally.
- The following SSM Parameter must exist in your AWS account as a `SecureString`:
  - `WANDB_API_KEY`

### Deployment Steps

1. Navigate to the infrastructure directory:

    ```bash
    cd wandb-mcp/infra
    ```

2. Deploy the infrastructure:

    ```bash
    uv run pulumi up --stack labs
    ```

    Pulumi will build the Docker image, push it to a private ECR repository, and deploy the service to ECS Fargate. The output will include the public URL of the load balancer.

    **Note for MacBook (ARM-based) users:** The Docker image is built for the `linux/amd64` platform to ensure compatibility with AWS Fargate. Docker Desktop handles this cross-platform build automatically.

### Destroying the Infrastructure

To tear down all the created AWS resources, run:

```bash
uv run pulumi destroy --stack labs
```
