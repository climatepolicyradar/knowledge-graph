# Infrastructure-as-code

Pulumi manages a small amount of AWS infrastructure for this repo. There is one
stack per AWS environment, named after that environment.

## What's managed

- **ECR repository `knowledge-graph`** — exists in all four environments, and
  holds the images that the Prefect flow deployments run from. The repository
  name has to match the name of this GitHub repo, because that is where
  `.github/workflows/prefect_deploy.yml` gets it from.
  Each repository has a lifecycle policy that expires untagged images after 7
  days and keeps only the 50 most recently pushed tagged images.
- **S3 bucket `s3://cpr-kg-feather-files`** — production only, with cross-account
  read access for labs. Labs needs it because the feather files in this bucket
  are used both by model training (AWS production) and by the vibe checker (AWS
  labs).

Vibe checker infra on AWS labs is managed separately in `vibe-checker/infra`.
