# Vibe check

A web-based tool to help concept developers understand how classifiers might perform on real-world documents, before going through the formal classifier evaluation process.

The site displays predictions for each candidate classifier by highlighting the predicted text spans from each model. It also allows users to search the predicted passages and filter by region, translation status, and corpus type.

The generated site has the following structure:

```
dist/
├── index.html                # List of all concepts
├── static/                   # Static assets (CSS, JS, images)
└── {concept_id}/             # Directory for each concept
    ├── index.html            # Concept details and available classifiers
    ├── {classifier_id}.html  # Predictions for a specific classifier
    └── {classifier_id}.json  # Raw predictions in JSONL format
```

## Generating the site with new data

Before generating the site, ensure you have run the prediction pipeline:

```bash
just predict Q123  # Replacing Q123 with your concept ID
```

To generate a new set of HTML files with the latest predictions, run:

```bash
just generate-static-site vibe_check
```

The HTML files will be saved locally to the `static_sites/vibe_check/dist` directory.

To preview the site locally, you can start a local server:

```bash
just serve-static-site vibe_check
```

The site will be available at `http://localhost:8080`.

## Deployment

The site is deployed using AWS CloudFront and S3.

### Deploy the infrastructure

This step is only required if you're making changes to the infrastructure itself - skip it if you're just updating the site content, and move straight on to [Pushing new data to s3](#pushing-new-data-to-s3).

```bash
# First, navigate to the infrastructure directory
cd static_sites/vibe_check/infra

# Install dependencies
pip install -r requirements.txt

# Deploy and get the outputs
pulumi up --stack labs
```

The deployment creates:

- An S3 bucket (`cpr-knowledge-graph-vibe-check`) configured for static website hosting
- A CloudFront distribution providing:
  - HTTPS support with automatic HTTP to HTTPS redirection
  - Clean URLs (no index.html suffixes needed) via CloudFront Functions
  - Global CDN distribution (using Price Class 100 edge locations)
  - HTTP/2 and IPv6 support
  - Optimized caching using AWS's managed CachingOptimized policy
  - Limited to GET, HEAD, and OPTIONS methods for security
- Appropriate security policies and access controls via Origin Access Identity (OAI)

The infrastructure is defined using Pulumi in `infra/__main__.py`.

### Pushing new data to s3

Once the infrastructure is ready, you can deploy the static site files:

```bash
# get the bucket name from the pulumi stack outputs
export BUCKET_NAME=$(pulumi stack output bucket_name --stack labs)

# Sync the dist directory to S3, removing any old files
aws s3 sync dist "s3://$BUCKET_NAME" --profile=labs --delete

# The site will be available at the CloudFront URL, which you can get with:
pulumi stack output cloudfront_url --stack labs
```

### Tearing down the infrastructure

To tear everything down, make your way to the `infra` directory and run:

```bash
pulumi destroy --stack labs
```
