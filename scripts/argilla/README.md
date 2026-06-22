# Scripts for working with Argilla

## Datasets

- `push_new_dataset`: adds passages to a new dataset in Argilla
- `extend_existing_dataset`: adds *more* passages to a dataset in Argilla. Avoids duplicates being added to Argilla from the input JSONL, based on an exact text match.

Both require use of the sample script (`scripts/sample.py`). They're set to load in 130 passages by default, meaning that the output of the sample script can be larger.

## Users

- `users.py`: manage Argilla users. It has two commands:
  - `create`: interactively create a single user as an `annotator` or an `owner`. Labellers are also assigned to the `knowledge-graph` workspace.
  - `list`: print every Argilla user with their name, role and workspace memberships.

Argilla owner credentials are pulled from SSM (`/Argilla/APIURL` and `/Argilla/Owner/APIKey`), so you must be authenticated to AWS when running it. Run with `uv run python scripts/argilla/users.py create` or `... list`.
