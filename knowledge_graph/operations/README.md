# `knowledge_graph/operations/`

Canonical home for **what** an operation does — the reusable, Prefect-free,
CLI-free domain logic.

Each module here owns the core logic for one operation (evaluate, get-concept,
predict, …). These modules know nothing about Prefect orchestration (`flows/`) or
CLI entrypoints (`scripts/`):

- **`flows/`** wraps operations for Prefect orchestration.
- **`scripts/`** wraps operations as thin Typer CLI entrypoints (e.g. `just evaluate`).
- **`knowledge_graph.classifier.autollm`** imports operations directly.

## Modules

See each module's docstring for what it does:

- `build_dataset.py`
- `evaluate.py`
- `get_concept.py`
- `predict.py`
- `snowflake.py`

## Conventions

- Operations must live at or below `knowledge_graph/` so that library code
  (e.g. `autollm`) can import them without an import cycle — library code must
  never import *up* into `scripts/`.
- Use `knowledge_graph.utils.get_logger` (not `logging.getLogger`) so logs are
  picked up when an operation runs under Prefect.
- Prefer PEP 604 unions (`X | None`) over `typing.Optional`.
