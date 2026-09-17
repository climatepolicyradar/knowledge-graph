import typer
from rich.console import Console

from flows.classifier_specs.spec_interface import yaml_spec_to_json
from knowledge_graph.cloud import AwsEnv

app = typer.Typer()
console = Console()


@app.command()
def check_classifier_specs(
    aws_env: AwsEnv = typer.Argument(
        help="Which aws environment to look for results in. Determines which spec file"
        "to use",
        default=AwsEnv.production,
    ),
):
    specs = yaml_spec_to_json(aws_env=aws_env)
    console.print(specs)


if __name__ == "__main__":
    app()
