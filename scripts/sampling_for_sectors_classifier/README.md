# Sampling for sectors classifier

We're currently trying to build a classifier for [economic sectors](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q709). To build classifiers for the more nuanced subconcepts, we need to collect a set of hand-labelled passages of text which pertain to each concept. To build that hand-labelled dataset, we need data for labelling, and for that, we need to parse a whole load of new documents.

These scripts parse a set of docs from corporate disclosures and litigation so that they can be sampled by a separate script in the knowledge-graph repo.

The script uses the `azure_pdf_parser` cli runner to do this. The pdfs and parser output objects are written to a local `data/` directory, which is not committed to version control. Instead, the data is persisted in an s3 bucket for later use.

Then we use the `navigator_document_parser` library to translate the documents to English, and extract the text from the documents. The text is then saved to a local `data/` directory, which is also not committed to version control. Instead, the data is persisted in an s3 bucket for later use.

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

The sampler used in 

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

## Argilla workspace stuff

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
