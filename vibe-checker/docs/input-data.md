# Input data

The vibe checker's inference flow depends on three pre-built files stored in the S3 bucket:

```text
s3://{BUCKET_NAME}/
├── passages_dataset.feather          # Balanced sample of passages from the document corpus
├── passages_embeddings.npy           # Pre-computed embedding for each passage
└── passages_embeddings_metadata.json # Embedding model name and other metadata
```

The `vibe_check_inference` flow loads them automatically at the start of each run.

## How they're generated

These files are produced by the **`generate_vibe_checker_embeddings`** Prefect flow
(`flows/generate_vibe_checker_embeddings.py`), which runs on a schedule (monthly) and:

1. Reads a balanced evenly-sampled subset of the corpus (stratified across geographies, corpus types, and translation status) produced in S3 by the `build_dataset` flow (sourced from Snowflake).
2. Computes passage embeddings using the `EMBEDDING_MODEL` constant in the flow
   (`BAAI/bge-small-en-v1.5` by default).
3. Uploads the three files above to the vibe-checker S3 bucket.

The sample is broad and evenly-sampled enough to give a representative view of our corpus. Because the flow is scheduled, you don't normally need to do anything to keep the inputs fresh, and you shouldn't need to regenerate them when adding new concepts to the `config.yml` file.

## Regenerating on demand

To refresh the files outside the schedule (e.g. after the upstream corpus changes, or if the files are deleted/corrupted), trigger a run of the `generate_vibe_checker_embeddings` deployment from Prefect.

**Note**: if you change the embedding model constant in the flow, the new embeddings will be incompatible with any previously stored concept embeddings. The inference flow re-encodes concepts on every run, so this shouldn't be a problem, but it's worth noting explicitly.
