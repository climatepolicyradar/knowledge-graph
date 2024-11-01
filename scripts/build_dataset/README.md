# Create a combined dataset for sampling

We need to sample from datasets which haven't yet been parsed by the core pipeline (eg corporate disclosures, litigation documents, etc). The scripts in this directory will download and parse these datasets, and then combine them into a single dataset which can be used by separate processes for sampling.

The scripts are numbered according to the order in which they should be run.

- `01_download_corporate_disclosures.py`
- `02_download_litigation.py`
- `03_add_geography.py`
- `04_parse.py`
- `05_translate.py`
- `06_merge.py`

The final dataset will be saved in `data/processed/combined_dataset.feather`.
