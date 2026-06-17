"""
Reusable, Prefect-free domain logic for knowledge-graph operations.

Each module here is the single home for *what* an operation does (querying,
transforming, predicting). Prefect orchestration that wraps these lives in
`flows/`. Modules in this package must never import from `flows/`.
"""
