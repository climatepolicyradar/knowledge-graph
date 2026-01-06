# Vibe Checker Infrastructure

Pulumi IaC for deploying the vibe-checker webapp on AWS ECS Fargate.

## Setup

This infrastructure is deployed in the labs AWS account, so you'll need to authenticate with AWS to access it. Before anything else, you'll need to run:

```bash
aws sso login --profile labs
```

## Deployment

Before deploying the infrastructure, you can preview the changes that will be made by running:

```bash
pulumi preview
```

If the preview looks good, you can apply the changes by running:

```bash
pulumi up
```

and selecting `yes` when prompted.

## View Outputs

The stack has a few useful outputs which you can view by running:

```bash
pulumi stack output
```

You can export the outputs directly into your current shell environment by running:

```bash
export $(pulumi stack output --json | jq -r 'to_entries | .[] | "\(.key)=\(.value)"')
```

## Destroy

If you need to tear down the full stack, you can destroy it by running:

```bash
pulumi destroy
```

and selecting `yes` when prompted.
