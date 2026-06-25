# `knowledge_graph/operations/`

Canonical home for **what** an operation does — the reusable, Prefect-free,
CLI-free domain logic.

Each module here owns the core logic for one operation (evaluate, get-concept,
predict, …). These modules know nothing about Prefect orchestration (`flows/`) or
CLI entrypoints (`scripts/`):

- **`flows/`** wraps operations for Prefect orchestration.
- **`scripts/`** wraps operations as thin Typer CLI entrypoints (e.g. `just evaluate`).
- **`knowledge_graph.classifier.autollm`** imports operations directly.
