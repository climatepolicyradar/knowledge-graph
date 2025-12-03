# Vibe Checker

[The vibe checker](https://vibe-checker.labs.climatepolicyradar.org) is a tool that gives concept developers a qualitative feel for how classifiers might perform on real-world policy documents. In combination with the quantitative evaluation results from the formal labelling process (stored alongside the models in Weights & Biases), this should provide a full picture of a classifier's performance and readiness for deployment.

The vibe checker is composed of two main components:

- A Prefect flow which runs inference on a small, well-chosen sample of passages from our dataset.
- A webapp which displays the predictions of the classifiers, with UI for filtering, searching, and comparing the predictions.

The prefect flow runs nightly and stores predictions in an S3 bucket. The webapp is deployed on ECS Fargate and reads predictions directly from S3.

## Running the flow locally

`flows/vibe_check.py` contains a Prefect flow (`vibe_check_inference`) which runs the inference pipeline on a pre-determined set of passages from our dataset. You can run it locally using Prefect CLI.

By default, the flow will run on all concepts defined in the `vibe-checker/config.yml` file.

```bash
prefect flow run flows.vibe_check:vibe_check_inference
```

However, if you want to run the flow on a specific set of concepts, you can pass the Wikibase IDs as a parameter:

```bash
prefect flow run flows.vibe_check:vibe_check_inference --param wikibase_ids='["Q69","Q420"]'
```

Both options will dump the results of the inference to the s3 bucket, in the `{concept_id}/{classifier_id}/` directory, where they'll immediately be available to users via the webapp.

## S3 bucket structure

The S3 bucket stores both input datasets and output predictions. The structure is as follows:

```text
s3://{BUCKET_NAME}/
├── passages_dataset.feather        # Input: Passages dataset
├── passages_embeddings.npy         # Input: Pre-computed embeddings used to sample potentially relevant passages for a given concept
├── passages_embeddings_metadata.json # Input: Metadata about the embeddings model used to compute the embeddings
├── {concept_id}/{classifier_id}/
│   ├── predictions.jsonl           # Output: All predictions for the given concept and classifier, with one prediction per line. Can contain negatives as well as positives.
│   ├── concept.json                # Output: A full copy of the concept metadata from Wikibase at the time of the inference
│   └── classifier.json             # Output: Metadata about the classifier used to generate the predictions
```

## Changing the list of default concepts to process

To update the config file, you should edit the config file in `vibe-checker/config.yml`. The file should be a list of Wikibase IDs:

```yaml
- Q123
- Q456
- Q789
```

## Further instructions

Take a look at the [webapp](./webapp/README.md) and [infra](./infra/README.md) directories for more detailed instructions on how to deploy and run the webapp and the associated infrastructure.
