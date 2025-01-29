# Vibe check

A web-based tool for visualizing and exploring predictions from our candidate classifiers. It helps concept developers understand how our classifiers perform on real-world documents.

The tool takes predictions from our classifier pipeline and generates a static website which:

- Lists all concepts with available classifiers
- Shows detailed predictions for each classifier, with highlighted text spans
- Provides filtering by region, translation status, and corpus type
- Includes search functionality for finding specific passages
- Supports both light and dark modes for comfortable viewing

## Usage

### Prerequisites

Before generating the site, ensure you have:

1. Run predictions for at least one concept (they should be in `data/processed/predictions/`)
2. The corresponding classifier models (in `data/processed/classifiers/`)

### Generating the Site

To generate a new set of HTML files with the latest predictions, run:

```bash
poetry run python -m vibe_check
```

The HTML files will be saved to the `data/build/vibe_check` directory.

To preview the site locally, you can use Python's built-in HTTP server:

```bash
python -m http.server --directory data/build/vibe_check
```

The site will be available at `http://localhost:8000`.

### Adding New Data

To generate predictions for a new concept:

1. Run the prediction pipeline:

```bash
just predict Q123  # Replace Q123 with your concept ID
```

2. Generate a fresh copy of the site:

```bash
poetry run python -m vibe_check
```

## Deployment

The site is deployed using AWS CloudFront and S3, giving us HTTPS support and clean URLs. Here's how to deploy:

### 1. Ensure you're using the correct AWS account

```bash
# Configure Pulumi to use the labs AWS profile
pulumi config set aws:profile labs

# Verify you're using the correct AWS account
aws sts get-caller-identity --profile=labs
```

### 2. Deploy the infrastructure

```bash
# Navigate to the infrastructure directory
cd uis/vibe_check/infra

# Install dependencies
pip install -r requirements.txt

# Deploy and get the outputs
pulumi up --stack labs

# The command will output your CloudFront URL and bucket name
```

### 3. Deploy the site content

Once the infrastructure is ready, you can deploy the static site files:

```bash
# export the bucket name
export BUCKET_NAME=$(pulumi stack output bucket_name --stack labs)

# Sync the build directory to S3, removing any old files
aws s3 sync data/build/vibe_check "s3://$BUCKET_NAME" --profile=labs --delete

# The site will be available at the CloudFront URL (which you can get with)
pulumi stack output cloudfront_url --stack labs
```

### Infrastructure Details

The deployment creates:

- An S3 bucket configured for static website hosting
- A CloudFront distribution providing:
  - HTTPS support
  - Clean URLs (no .html suffixes needed)
  - Global CDN distribution
  - Automatic HTTP to HTTPS redirection
- Appropriate security policies and access controls

The infrastructure is defined in `infra/__main__.py` using Pulumi.

### Updating the Site

When you have new predictions to publish:

1. Generate new predictions:

```bash
just predict Q123  # Replace Q123 with your concept ID
```

2. Rebuild the site:

```bash
poetry run python -m vibe_check
```

3. Deploy the updates:

```bash
# export the bucket name
export BUCKET_NAME=$(pulumi stack output bucket_name --stack labs)

aws s3 sync data/build/vibe_check "s3://$BUCKET_NAME" --profile=labs --delete
```

### Tearing Down

To remove the infrastructure:

```bash
cd uis/vibe_check/infra
pulumi destroy --stack labs
```

## Site Structure

The generated site follows this structure:

```
data/build/vibe_check/
├── index.html                # List of all concepts
├── static/                   # Static assets (CSS, JS, images)
└── {concept_id}/             # Directory for each concept
    ├── index.html            # Concept details and available classifiers
    └── {classifier_id}.html  # Predictions for a specific classifier
```
