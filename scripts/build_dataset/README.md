# Create a combined dataset for sampling

We need to sample from datasets which haven't yet been parsed by the core pipeline (eg corporate disclosures, litigation documents, etc). The scripts in this directory will download and parse these datasets, and then combine them into a single dataset which can be used by separate processes for sampling.

The scripts are numbered according to the order in which they should be run. You can run all of the scripts in sequence with `just build-dataset`.

The final dataset will be saved in `data/processed/combined_dataset.feather`.

**NOTE:** The final step in this process filters out any passages which are less than 20 characters long from the dataset. In practice, we've found that such short passages are too short to be useful for evaluation.  
We're also making a small assumption here that upcoming work on our text chunking process will make evaluation of such short, semantically empty passages redundant, as the future version will produce them far less often.
