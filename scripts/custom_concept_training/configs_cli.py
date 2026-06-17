from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel
from rich.console import Console

from knowledge_graph.custom_classifier_config import CustomClassifierConfig
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.custom_concept_training.validate import (
    CONFIG_DIR,
    check_wikibase_ids,
    validate_dir,
    validate_file,
)

app = typer.Typer()
console = Console()


# --- readable YAML output (literal block style for multiline strings) ---
class _LiteralDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _str_representer)


@app.command()
def validate(
    config_path: Path | None = typer.Argument(
        None, help="Validate a single config file. Defaults to all configs in the dir."
    ),
    check_all_configs: bool = typer.Option(
        False, help="Validate every config in the dir (incl. duplicate-concept check)."
    ),
    check_wikibase: bool = typer.Option(
        False,
        help="Also confirm every wikibase_id/related definition resolves.",
    ),
    config_dir: Path = CONFIG_DIR,
):
    """Validate a single config file, or every config in the dir."""
    try:
        if config_path is not None:
            paths = [config_path]
            validate_file(config_path)
        elif check_all_configs:
            paths = sorted(config_dir.glob("*.yaml"))
            validate_dir(config_dir)
        else:
            raise typer.BadParameter("Pass a config file path or --check-all-configs.")

        if check_wikibase:
            session = WikibaseSession()
            for p in paths:
                check_wikibase_ids(CustomClassifierConfig.from_yaml(p), session)
    except ValueError as e:
        console.print(f"[red]FAIL[/red] {e}")
        raise typer.Exit(code=1)

    console.print("[green]All config(s) valid.[/green]")


def _example_value(field_name: str, field_info: Any) -> Any:
    """Example for a required field with no default/default_factory."""
    examples = {
        "wikibase_id": "Q123",
        "model_name": "openrouter:openai/gpt-5",
    }
    return examples.get(field_name, f"<{field_name}>")


def to_example_dict(model_class: type[BaseModel]) -> dict[str, Any]:
    """Build a template dict from the Pydantic BaseModel."""
    result: dict[str, Any] = {}
    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            result[field_name] = to_example_dict(annotation)
        elif field_info.is_required():
            result[field_name] = _example_value(field_name, field_info)
        else:
            result[field_name] = field_info.get_default(call_default_factory=True)
    return result


@app.command()
def create(
    wikibase_id: str = typer.Argument(..., help="Concept Wikibase ID, e.g. Q123"),
):
    """Write a template config YAML for a concept to CONFIG_DIR for user to fill in."""
    wid = WikibaseID(wikibase_id)
    data = to_example_dict(CustomClassifierConfig)
    data["wikibase_id"] = str(wid)
    out = CONFIG_DIR / f"q{wid.numeric}.yaml"
    out.write_text(
        yaml.dump(
            data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=4096
        )
    )
    console.print(
        f"[green]Wrote template to {out}.[/green] Fill in llm.model_name etc., then run "
        f"`uv run classifier-configs validate {out}`."
    )


if __name__ == "__main__":
    app()
