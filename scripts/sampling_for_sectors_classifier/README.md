# Sampling for sectors classifier

We're currently trying to build a classifier for [economic sectors](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q709). To evaluate the performance of these classifiers (and to build more sophisticated classifiers for the more nuanced concepts), we need to collect a set of hand-labelled passages of text which pertain to each concept. To build those hand-labelled datasets, we need data for labelling, and for that, we need to parse a whole load of new documents.

These scripts parse a set of docs from new sources (MCFs, litigation, corporate disclosures) so they can be combined with other datasets and sampled for hand-labelling.

After downloading the raw pdf documents, they're parsed by the `azure_pdf_parser` cli runner. Then we use the `navigator_document_parser` library to translate the documents to English, and extract the text from the documents.

We also add fields for world bank region and dataset source, so that we can produce a weighted sample of documents to satisfy our equity constraints.

The data is persisted in an s3 bucket for later use.

## Environment variables

The `translate_docs.py` script runs document translation using the google cloud translate API.

To get a set of CPR's google API credentials, clone the `navigator-data-pipeline` repo.Then, run the following command to fetch and save the google credentials:

```sh
pulumi config get pipeline:google_creds_json | base64 --decode > google-credentials.json
```

The response from pulumi will be a base64 encoded string. We decode this, and pipe the output to a file called `google-credentials.json`.

Set a `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of this file in your `.env` file:

```sh
GOOGLE_APPLICATION_CREDENTIALS=google-credentials.json
```

## Sampling config

Some of the later scripts in this directory require a sampling config object which is read from a local yaml file. This file should look something like this:

```yaml
stratified_columns: ["world_bank_region", "dataset_name"]
equal_columns: ["translated"]
sample_size: 130
negative_proportion: 0.2
wikibase_ids: ["Q123", "Q456", "Q789"]
labellers: ["person1", "person2", "person3"]
```

## Running the scripts

In rough order of execution:

- `scrape_green_climate_funds.py`
- `download_green_climate_fund_docs.py`
- `download_litigation_docs.py`
- `parse_docs.py`
- `translate_docs.py`
- `add_geography_to_litigation.py`
- `push_to_s3.py` (optional)
- `combine_datasets.py`
- `training.py`
- `inference.py`
- `sample_passages.py`
- `push_sampled_passages_to_argilla.py`

## Argilla workspace management

To create a user in Argilla from the CLI, first login:

```sh
argilla login --api-url WHATEVER_THE_API_URL_IS
```

You'll be asked for your API key, which can be found in the settings of the Argilla web app. Once you're logged in, you can run:

```sh
argilla users create --username THEIR_NAME --password SOMETHING_SUPER_SECURE_AND_SECRET --first-name THEIR_FIRST_NAME --last-name THEIR_FIRST_NAME --role annotator
```

To add the user to their corresponding workspace, run:

```sh
argilla workspaces --name THEIR_NAME add-user THEIR_NAME
```

## Pushing to s3

To push the data to s3, you'll need a set of credentials in your `~/.aws`directory which have access to the `labs` account. To login and refresh your credentials, run

```sh
aws sso login --profile=labs
```

Then, you can run the `push_to_s3.py` script to push the data to s3.

To _pull_ the data from s3, you can run the following command:

```sh
aws s3 sync s3://cpr-sectors-classifier-sampling/ ./data --profile=labs
```

## Pushing a new set of concepts to argilla

To push a new set of concepts to argilla (assuming all of the steps have been run once before), you should:

1. Add a new `YOUR_TAXONOMY_NAME.yaml` file to the `scripts/sampling_for_sectors_classifier/config` directory following the format of the existing files.
2. Change the config file named in `training.py`, `inference.py`, `sample_passages.py`, and `push_sampled_passages_to_argilla.py` to the new file.
3. Run the `training.py` script to train and save a set of new models.
4. Run the `inference.py` script to generate predictions for the new set of concepts on the full `combined_dataset.feather` dataset
5. Run the `sample_passages.py` script to sample a set of passages which will be used for hand-labelling.
6. Run the `push_sampled_passages_to_argilla.py` script to push the sampled passages to argilla for labelling.
