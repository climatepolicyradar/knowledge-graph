# Vibe Checker

The [vibe checker](https://vibe-checker.labs.climatepolicyradar.org) lets us quickly get a sense of how well our classifiers are performing on real-world policy documents. For each concept, it samples passages that are semantically similar to the concept, runs classifier inference on them, and makes the results available via a webapp.

## Components

- **`flows/vibe_check.py`** — A Prefect flow that runs inference and pushes results to S3
- **`vibe-checker/webapp/`** — A Next.js app that reads results from S3 and displays them

## Docs

- [Adding concepts to the standard inference flow](docs/adding-concepts.md)
- [Uploading results from a custom classifier](docs/custom-classifiers.md)
- [Webapp development and deployment](webapp/README.md)
- [Infrastructure](infra/README.md) and [architecture](infra/architecture.md)
