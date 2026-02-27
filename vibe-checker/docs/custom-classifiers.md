# Uploading results from a custom classifier

If you want to vibe check a custom or experimental classifier, you can produce and upload the output files directly without going through the Prefect flow. The new classifier will appear automatically in the webapp's classifier dropdown for that concept once the files are in S3.

## Producing the output files

Assuming you have a classifier that implements `.predict(texts) -> list[list[Span]]`, you should be able to adapt the example in [`custom_concept_demo.py`](../scripts/custom_concept_demo.py) to get started. It should walk you through the structures that you eventually need to push to S3.

There are three major components that need to be uploaded for the new classifier to appear successfully in the webapp:

- metadata about the concept. This should just be a JSON dump of the `Concept` object.
- metadata about the classifier. This should have three fields: `id` (the classifier's canonical/deterministic ID), `name` (the classifier's classname, e.g. `KeywordClassifier`), and `date` (the date that the classifier was created)
- the predictions, stored as a JSONL file with each line being a JSON dump of the `LabelledPassageWithMarkup` object. The important thing here is that the json objects should have a markup field, where spans are highlighted with `<span class="prediction-highlight">` tags. These spans will then be highlighted using the webapp's CSS.

The webapp uses the following metadata fields for filtering, so they also need to be present on each one of the labelled passages in `predictions.jsonl`. These should all be stringified values, matching what the standard flow produces from the passages dataset.

| Field | Example |
|---|---|
| `document_id` | `CCLW.executive.12345.6789` |
| `translated` | `True` or `False` |
| `document_metadata.corpus_type_name` | `UNFCCC` |
| `document_metadata.publication_ts` | `2020-01-01 00:00:00` |
| `document_metadata.slug` | document slug |
| `document_metadata.family_slug` | family slug |
| `text_block.text_block_id` | passage identifier |
| `text_block.page_number` | page number |
| `world_bank_region` | `Sub-Saharan Africa` |
