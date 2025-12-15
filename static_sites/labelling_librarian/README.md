# Labelling Librarian

A web-based tool that helps concept developers maintain the quality and consistency of our [labelled data](https://argilla.labs.climatepolicyradar.org/).

The tool checks for issues with labelled datasets in the Argilla and generates a static website which:

## Usage

### Generating the static site

To check for issues and generate a new version of the static site, run:

```bash
just generate-static-site labelling_librarian
```

The HTML files will be saved locally to the `static_sites/labelling_librarian/dist` directory.

To preview the site locally, you can start a local server:

```bash
just serve-static-site labelling_librarian
```

The site will be available at `http://localhost:8080`.

## Deployment

The site is deployed using AWS CloudFront and S3.

### Deploy the infrastructure

This step is only required if you're making changes to the infrastructure itself - skip it if you're just updating the site content, and move straight on to [Pushing new data to s3](#pushing-new-data-to-s3).

```bash
# First, navigate to the infrastructure directory
cd static_sites/labelling_librarian/infra

# Install dependencies
pip install -r requirements.txt

# Deploy and get the outputs
pulumi up --stack labs
```

The deployment creates:

- An S3 bucket (`cpr-knowledge-graph-concept-librarian`) configured for static website hosting
- A CloudFront distribution providing:
  - HTTPS support with automatic HTTP to HTTPS redirection
  - Clean URLs (no index.html suffixes needed) via CloudFront Functions
  - Global CDN distribution (using Price Class 100 edge locations)
  - HTTP/2 and IPv6 support
  - Optimized caching
  - Limited to GET, HEAD, and OPTIONS methods for security
- Appropriate security policies and access controls via Origin Access Identity (OAI)

The infrastructure is defined using Pulumi in `infra/__main__.py`.

### Pushing new data to s3

Once the infrastructure is ready, you can deploy the static site files:

```bash
# First, navigate to the concept librarian directory
cd static_sites/labelling_librarian

# get the bucket name from the pulumi stack outputs
export BUCKET_NAME=$(cd infra && pulumi stack output bucket_name --stack labs)

# Sync the dist directory to S3, removing any old files
aws s3 sync dist "s3://$BUCKET_NAME/dist" --profile=labs --delete

# The site will be available at the CloudFront URL, which you can get with:
cd infra && pulumi stack output cloudfront_url --stack labs
```

### Tearing down the infrastructure

To tear everything down, make your way to the `infra` directory and run:

```bash
pulumi destroy --stack labs
```
