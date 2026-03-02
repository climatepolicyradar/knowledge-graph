# Vibe Checker

The [vibe checker](https://vibe-checker.labs.climatepolicyradar.org) lets us quickly get a sense of how well our classifiers are performing on real-world policy documents. For each concept, we sample a set of passages which are semantically similar to the concept, run classifier inference on them, and make the results available via a webapp. Our domain experts use this to get a sense of how the classifiers behave in the wild, and to identify any potential issues.

## Components

- `flows/vibe_check.py` - A Prefect flow that runs inference and pushes results to S3
- `vibe-checker/webapp/` - A Next.js app that reads results from S3 and displays them
- `vibe-checker/infra/` - Pulumi IaC for deploying the s3 bucket and the webapp on AWS ECS Fargate

### S3 bucket structure

The vibe checker's inference flow and webapp depend on a set of files stored in the S3 bucket, which is managed by the pulumi IaC. The structure of the bucket is as follows:

```text
s3://{BUCKET_NAME}/
├── passages_dataset.feather              # Input: passages dataset
├── passages_embeddings.npy               # Input: pre-computed passage embeddings
├── passages_embeddings_metadata.json     # Input: metadata about the embedding model
└── {concept_id}/{classifier_id}/
    ├── predictions.jsonl                 # Output: labelled passages (positive and negative)
    ├── concept.json                      # Output: concept metadata at inference time
    └── classifier.json                   # Output: classifier metadata (id, name, date)
```

You can get the name of the bucket from the pulumi stack output:

```bash
pulumi stack output bucket_name
```

## More documentation

- [Adding new concepts to the regular inference schedule](docs/adding-concepts.md)
- [Uploading results from a custom classifier](docs/custom-classifiers.md)
- [The passages feather file and the pre-computed embeddings](docs/input-data.md)
- [Webapp development and deployment](webapp/README.md)
- [The system's basic architecture](docs/architecture.md) and [how to work with the infrastructure](infra/README.md)

There are a few additional scripts in the [scripts](scripts/) directory which should illustrate how the data was put together, or how to extend the vibe-checker for custom results.
