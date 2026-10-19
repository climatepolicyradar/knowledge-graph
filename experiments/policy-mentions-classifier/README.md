# Policy mentions classifier

Finds mentions of policy documents (laws, strategies, plans, reports) in documents
such as NDCs and BTRs, and produces a workbook for checking whether each mentioned
document is already in CCLW.

The pipeline has three stages:

1. **BERT passage filter** (`predict.py`). A fine-tuned
   [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) binary
   classifier scores every passage for whether it mentions a document.
2. **Filter and de-duplicate** (`mentions.py filter`). Keep passages above the
   threshold, and collapse repeated boilerplate within each document.
3. **LLM span extraction** (`mentions.py extract`). An LLM marks up each mention,
   producing one row per mention with its extracted title in `labelled_text`.

`deliverable.py` then flags exclusions and writes the workbook.

The BERT weights live in `s3://policy-mentions-classifier/models` and are downloaded into
`./models/` on first run. If S3 access fails, unzip a copy of `models.zip` so that
`models/model.safetensors` sits at the top level of that folder.

This is a self-contained experiment. It does **not** follow the
`knowledge_graph/classifier/` patterns.

## Setup

```sh
just install-transformers   # one-time: installs the torch + transformers extra
```

- **Snowflake.** Passages are read from Snowflake via the `local_dev` connection in
  `~/.snowflake/connections.toml` (or `config.toml`), as in `scripts/build_dataset.py`.
- **LLM.** The default model is `openrouter:openai/gpt-4.1-mini`. If
  `OPENROUTER_API_KEY` isn't set, `mentions.py extract` reads it from SSM
  (`/OpenRouter/KGApiKey`) in the `--aws-env` account, which defaults to `labs`.
- **Workbook.** `deliverable.py` needs `openpyxl`, which isn't a project dependency;
  the command below adds it with `uv run --with`.

## End-to-end run

Put the document IDs in a file, one per line (`#` comments allowed). See
`inputs/fus-609.txt` for an example. Then:

```sh
IDS=experiments/policy-mentions-classifier/inputs/fus-609.txt
RUN=data/policy-mentions-classifier/fus-609

# 1. BERT over every English passage (>20 chars) of every document
uv run python experiments/policy-mentions-classifier/predict.py \
    --ids-file $IDS --combined-output $RUN/bert.parquet

# 2. keep class_1_prob >= 0.5 and de-duplicate within each document
uv run python experiments/policy-mentions-classifier/mentions.py filter \
    --bert $RUN/bert.parquet --run-dir $RUN

# 3. estimate cost, eyeball a 50-passage sample, then run everything
uv run python experiments/policy-mentions-classifier/mentions.py estimate --run-dir $RUN
uv run python experiments/policy-mentions-classifier/mentions.py extract --run-dir $RUN --limit 50
uv run python experiments/policy-mentions-classifier/mentions.py extract --run-dir $RUN

# 4. exclusion flags and workbook
uv run --with openpyxl python experiments/policy-mentions-classifier/deliverable.py \
    --run-dir $RUN --bert $RUN/bert.parquet --ids-file $IDS
```

`extract` caches LLM responses per passage in `$RUN/llm_cache.jsonl`, so re-running
resumes where the last run stopped and reuses the sample's results. API failures
aren't cached, so re-running also retries them.

You can still classify ad-hoc documents with `--document-id <id>` or
`--document-slug <slug>` (both repeatable). `predict.py` also writes one CSV per
document to `data/policy-mentions-classifier/outputs/<slug>.csv`.

## Outputs

`deliverable.py` writes `BTR_NDC_mentions_<YYYY-MM-DD>.xlsx` with these tabs:

1. **mentions.** One row per mention where neither exclude flag is set.
2. **excluded mentions.** Flagged rows, kept so false exclusions can be restored.
3. **duplicate rows.** Passages removed by de-duplication, with the passage that
   represents them and what it yielded.
4. **criteria.** Threshold, model, keyword list, matching rule and run date.
5. **README.** Annex decisions, documents with zero passages or mentions, and
   per-document counts.

Columns in the mentions and excluded tabs, in order:

| column | content |
|---|---|
| `geography` | country name, from the CPR geographies API |
| `labelled_text` | the extracted title of the mentioned document |
| `relevant`, `duplicate CCLW`, `notes` | empty, for reviewers to fill in |
| `text` | the passage, with HTML entities unescaped |
| `page_number` | first page of the passage, 1-indexed |
| `document_id`, `document_name`, `document_type` | source document; type is NDC or BTR |
| `document_source_url` | original URL of the source document |
| `keyword_exclude` | `labelled_text` contains an excluded keyword as a whole word |
| `nonalphabetic_exclude` | `labelled_text` has no alphabetic characters |
| `search_link` | `app.climatepolicyradar.org/search?q=<labelled_text>` |
| `search_link_with_geography` | the same, filtered by the geography's slug (`&l=united-kingdom`) |

Exclusion keywords match case-insensitively, and only where the keyword isn't
directly preceded or followed by a letter (a plural "s" is allowed). So `table`
doesn't match "sustainable", but `ndc` matches "NDC3". The keyword list in
`deliverable.py` mirrors the Notion page "BTR Update keywords to exclude", which is
canonical.

Keywords are applied to `labelled_text` after extraction, not to passages before the
LLM: most NDC and BTR passages contain a keyword somewhere.

## Threshold

`predict.py` marks `predicted_class = 1` when `class_1_prob >= 0.5`. BERT is only a
cost filter ahead of the LLM, so the threshold favours recall. `class_1_prob` is
always kept, so `mentions.py filter --threshold` can change it without re-running
BERT.

## Notes

- Passages come from `PRODUCTION.PUBLISHED.PASSAGES` in reading order (`IDX`).
  `passage_id` is the table's `ID`, `text_block_id` is the `IDX`, and `page_numbers`
  is `PAGE_NUMBERS`, which is 0-indexed.
- Device is chosen automatically: CUDA, then Apple Silicon MPS, then CPU. If an op
  is unsupported on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Passages are truncated to 512 tokens at inference, matching training.
- `labelled_text` is always a substring of `text`. When the LLM alters the text it
  tags, the span is fuzzily re-aligned and `llm_exact_match` is False;
  `deliverable.py` prints how many.
