# Policy mentions classifier

A fine-tuned [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
binary sequence classifier that flags passages mentioning policy documents.

Ported from the (now archived) [policy-mentions-classifier repo](https://github.com/climatepolicyradar/policy-mentions-classifier). Only the
inference path was kept. The trained weights live in
`s3://policy-mentions-classifier/models` and are downloaded automatically into
`./models/` on first run.

This is a self-contained experiment — it does **not** follow the
`knowledge_graph/classifier/` patterns.

## Setup

```sh
just install-transformers   # one-time: installs the torch + transformers extra
```

Snowflake access is required to load documents. Locally this uses the
`local_dev` connection from `~/.snowflake/config.toml` (same as
`scripts/build_dataset.py`).

## Usage

Documents are selected by ID or slug; all English passages for each document are
pulled from Snowflake, classified, and written to one CSV per document at
`data/policy-mentions-classifier/outputs/<slug>.csv`.

```sh
# by document id (repeatable)
uv run python experiments/policy-mentions-classifier/predict.py --document-id <id> --document-id <id>

# by document slug (repeatable)
uv run python experiments/policy-mentions-classifier/predict.py --document-slug <slug>
```

Output columns: `text`, `text_block_id`, `document_id`, `document_name`,
`document_slug`, `predicted_class`, `confidence`, `class_0_prob`, `class_1_prob`.

## Notes

- `predicted_class` is 1 (policy mention) only when the model's class-1
  probability is ≥ 0.75 (the `THRESHOLD` constant), otherwise 0. `class_0_prob`
  / `class_1_prob` are the raw softmax probabilities; `confidence` is the
  probability of the assigned class.
- Device is chosen automatically: CUDA → Apple Silicon MPS → CPU. If an op is
  ever unsupported on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back
  to CPU for that op.
- Passages are truncated to 512 tokens at inference, matching how the model was
  used when trained.
- The Snowflake passages table (per `scripts/build_dataset.py`) has no native
  passage id, so `text_block_id` is a sequential index within each document.
- The model auto-downloads from S3 on first run and is reused from `./models/`
  thereafter.
