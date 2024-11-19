# Create a combined dataset for sampling

We need to sample from datasets which haven't yet been parsed by the core pipeline (eg corporate disclosures, litigation documents, etc). The scripts in this directory will download and parse these datasets, and then combine them into a single dataset which can be used by separate processes for sampling.

The scripts are numbered according to the order in which they should be run. You can run all of the scripts in sequence with `just build-dataset`.

The final dataset will be saved in `data/processed/combined_dataset.feather`.
