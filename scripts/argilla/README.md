# Scripts for working with Argilla

There are two scripts in this folder:

- `push_new_dataset`: adds passages to a new dataset in Argilla
- `extend_existing_dataset`: adds *more* passages to a dataset in Argilla. Avoids duplicates being added to Argilla from the input JSONL, based on an exact text match.

Both require use of the sample script (`scripts/sample.py`). They're set to load in 130 passages by default, meaning that the output of the sample script can be larger.
