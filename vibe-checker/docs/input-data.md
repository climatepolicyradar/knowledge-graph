# Input data

The vibe checker's inference flow depends on three pre-built files stored in the S3 bucket:

```text
s3://{BUCKET_NAME}/
├── passages_dataset.feather          # Sampled passages from the document corpus
├── passages_embeddings.npy           # Pre-computed embedding for each passage
└── passages_embeddings_metadata.json # Embedding model name and other metadata
```

The flow loads them automatically at the start of each run. The dataset should be broad and evenly-sampled enough to give a representative view of our corpus, so you shouldn't need to regenerate them when adding new concepts to the `config.yml` file.

## When you might need to regenerate them

- The upstream HuggingFace dataset (`ClimatePolicyRadar/all-document-text-data-weekly`) is updated and you want the vibe checker to reflect the latest documents
- The files are accidentally deleted or corrupted

## How to regenerate

The process is two steps: first build a sampled dataset, then compute and upload embeddings.

**Step 1**: Run `scripts/build_dataset.py` to download and sample the main CPR corpus. From the root of the repo, run:

```bash
uv run scripts/build_dataset.py
```

This produces `data/processed/sampled_dataset.feather` — a balanced sample across geographies, corpus types, and translation status. See that script's `--help` for more information.

**Step 2**: Run `vibe-checker/scripts/generate_passage_embeddings.py` with the feather file as input, eg:

```bash
uv run vibe-checker/scripts/generate_passage_embeddings.py data/processed/sampled_dataset.feather
```

This will:

1. Load the dataset from step 1
2. Compute passage embeddings using the `EMBEDDING_MODEL` constant in the script (`BAAI/bge-small-en-v1.5` by default)
3. Save the three output files locally (alongside the input feather)
4. Ask for confirmation before uploading to S3

**Note**: if you change the embedding model constant in the script, the new embeddings will be incompatible with any previously stored concept embeddings. The inference flow re-encodes concepts on every run, so this shouldn't be a problem, but it's worth noting explicitly.
