"""
CLI wrapper for refreshing classifier spec files.

The reusable logic lives in `flows.update_classifier_spec`; this module only adds the
Typer command used by the `update-inference-classifiers` console script.
"""

import typer
from dotenv import load_dotenv

from flows.update_classifier_spec import refresh_all_available_classifiers
from knowledge_graph.cloud import AwsEnv

load_dotenv()

app = typer.Typer()


@app.command()
def refresh(aws_envs: list[AwsEnv] | None = None) -> None:
    """Refreshes the classifier specs with the latest state of wandb."""
    refresh_all_available_classifiers(aws_envs=aws_envs)


if __name__ == "__main__":
    app()
