# Adding concepts to the standard inference flow

## Editing the concept list

The inference flow reads its list of concepts from `vibe-checker/config.yml`:

```yaml
- Q123
- Q456
- Q789
```

To add a new concept, add its Wikibase ID to `config.yml` and commit the change. When merged to main, the flow will be re-deployed, and the new concept will be added to the inference schedule.

To remove a concept from the inference schedule, delete its ID from `config.yml`. Its existing results will remain in S3 and will continue to be accessible in the webapp until manually deleted.

## Running inference

The `vibe_check_inference` flow runs automatically on Monday to Thursday at 8am. You can also trigger it manually from the Prefect UI.

To run inference on a specific set of concepts without modifying `config.yml`, you can trigger the `vibe_check_inference` flow with a `wikibase_ids` parameter via the Prefect UI.

The flow:

1. Loads the passages dataset and pre-computed embeddings from S3
2. For each concept, selects a reasonably-sized set of the most semantically similar passages
3. Gets or trains the concept's classifier via W&B using `run_training()` (fetches an existing model if one exists, otherwise trains a new one)
4. Runs inference and pushes the results to S3

Results are stored under `{concept_id}/{classifier_id}/` in the S3 bucket and are immediately available in the webapp. The webapp discovers which concepts to display by reading the S3 folder structure directly — any folder matching a Wikibase ID pattern (`Q` followed by digits) will appear automatically.

## S3 bucket structure

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
