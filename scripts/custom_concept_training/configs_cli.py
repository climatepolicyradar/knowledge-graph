import string
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from knowledge_graph.classifier.large_language_model import DEFAULT_SYSTEM_PROMPT
from knowledge_graph.custom_classifier_config import (
    DEFINITION_WIKIBASE_LENGTH_LIMIT,
    DESCRIPTION_WIKIBASE_LENGTH_LIMIT,
    SUPPORTED_LLM_MODELS,
    ConceptOverrides,
    CustomClassifierConfig,
    LLMClassifierConfig,
    SamplingConfig,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import ConceptNotFoundError, WikibaseSession
from scripts.custom_concept_training.validate import (
    CONFIG_DIR,
    check_wikibase_ids,
    validate_dir,
    validate_file,
)

app = typer.Typer()
console = Console()

TEMPLATE_PATH = Path(__file__).parent / "templates" / "labelling_guidelines.md"


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
        False,
        help="Also confirm every wikibase_id/related definition resolves (needs creds).",
    ),
    config_dir: Path = CONFIG_DIR,
):
    """
    Validate a single config file, or every config in the dir.

    The cross-file duplicate-concept check only runs when validating the whole dir (a single-file
    run can't see siblings). Per-file authoring feedback is also handled by `create`.
    """
    results = (
        {config_path: validate_file(config_path)}
        if config_path is not None
        else validate_dir(config_dir)
    )

    if check_all_configs:
        session = WikibaseSession()
        for path, errors in results.items():
            if errors:
                continue  # skip live check on files that don't parse
            cfg = CustomClassifierConfig.from_yaml(path)
            errors.extend(check_wikibase_ids(cfg, session))

    ok = True
    for path, errors in results.items():
        if errors:
            ok = False
            console.print(f"[red]FAIL[/red] {path.name}")
            for e in errors:
                console.print(f"    - {e}")
        else:
            console.print(f"[green]PASS[/green] {path.name}")

    if not ok:
        raise typer.Exit(code=1)
    console.print(f"\n[green]All {len(results)} config(s) valid.[/green]")


def _fetch_and_confirm(session: WikibaseSession, prompt: str) -> tuple[WikibaseID, str]:
    """Prompt for a WikibaseID, fetch it live, show its label, and confirm. Returns (id, label)."""
    while True:
        raw = typer.prompt(prompt).strip()
        try:
            wid = WikibaseID(raw)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            continue
        try:
            concept = session.get_concept(wid)
        except ConceptNotFoundError:
            console.print(f"[red]{wid} not found in Wikibase.[/red]")
            continue
        console.print(f"[green]{wid}[/green]: [bold]{concept.preferred_label}[/bold]")
        console.print((concept.definition or "(no definition)")[:500])
        if typer.confirm("Is this correct?", default=True):
            return wid, concept.preferred_label


@app.command()
def create(config_dir: Path = CONFIG_DIR):
    """Interactively build a new custom-classifier config YAML."""
    session = WikibaseSession()

    # 1. target concept
    wid, _ = _fetch_and_confirm(
        session, "Wikibase ID of the concept to classify (e.g. Q123)"
    )

    # 2. concept_overrides: bespoke definition/description too long for concept store ONLY
    caps = {
        "definition": DEFINITION_WIKIBASE_LENGTH_LIMIT,
        "description": DESCRIPTION_WIKIBASE_LENGTH_LIMIT,
    }
    overrides: dict[str, str] = {}
    for field, cap in caps.items():
        if not typer.confirm(
            f"\nOverride the {field}? Only if it exceeds {cap} chars — "
            "otherwise update it in the concept store (Wikibase).",
            default=False,
        ):
            continue
        text = (
            typer.edit(f"# Paste the full {field} below, then save & close.\n") or ""
        ).strip()
        if len(text) <= cap:
            console.print(
                f"[yellow]{field} is {len(text)} chars (<= {cap}); it fits the concept store. "
                "Please add it in Wikibase instead — skipping this override.[/yellow]"
            )
            continue
        overrides[field] = text

    # 3. model
    models = sorted(SUPPORTED_LLM_MODELS)
    console.print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        console.print(f"  {i}. {m}")
    idx = typer.prompt("Choose a model number", type=int, default=1)
    model_name = models[idx - 1]

    # 4. sampling (optional override of defaults)
    sampling = SamplingConfig()
    if typer.confirm(
        "\nCustomise sampling (otherwise defaults are used)?", default=False
    ):
        sampling = SamplingConfig(
            dataset_name=typer.prompt(
                "dataset_name [balanced/combined]", default=sampling.dataset_name
            ),
            sample_size=typer.prompt(
                "sample_size", type=int, default=sampling.sample_size
            ),
            min_negative_proportion=typer.prompt(
                "min_negative_proportion",
                type=float,
                default=sampling.min_negative_proportion,
            ),
        )

    # 5. related definitions referenced in the guidelines
    related: list[WikibaseID] = []
    related_labels: dict[WikibaseID, str] = {}
    while typer.confirm(
        f"\nAdd a related concept definition? ({len(related)} added so far)",
        default=False,
    ):
        rel_wid, rel_label = _fetch_and_confirm(
            session, "Related Wikibase ID (e.g. Q456)"
        )
        related.append(rel_wid)
        related_labels[rel_wid] = rel_label

    # 6. labelling guidelines (template + editor). Placeholders must match `related` exactly.
    guidelines: str | None = None
    if typer.confirm(
        "\nWrite labelling guidelines now (opens your editor)?", default=True
    ):
        template = TEMPLATE_PATH.read_text()
        if related:
            slots = "\n".join(f"- {related_labels[wid]}: {{{wid}}}" for wid in related)
            template += f"\n## 5. RELATED DEFINITIONS\n{slots}\n"

        expected = {str(wid) for wid in related}
        draft = template
        while True:
            draft = typer.edit(draft) or draft
            found = {name for _, name, _, _ in string.Formatter().parse(draft) if name}
            extra, missing = found - expected, expected - found
            if not extra and not missing:
                guidelines = draft.strip() or None
                break
            if extra:
                console.print(
                    f"[red]Placeholders with no matching related definition: {sorted(extra)}. "
                    "Remove the braces, or add them as related definitions.[/red]"
                )
            if missing:
                console.print(
                    f"[red]Related definitions not referenced in the guidelines: {sorted(missing)}. "
                    "Add a {Qxxx} placeholder for each.[/red]"
                )
            if not typer.confirm("Re-open the editor to fix?", default=True):
                raise typer.Abort()

    # 7. system prompt. At the moment bespoke prompt MUST keep the {concept_description} placeholder.
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if typer.confirm(
        "\nWrite a bespoke system prompt? (otherwise the standard one is used)",
        default=False,
    ):
        while True:
            edited = typer.edit(system_prompt) or system_prompt
            if "{concept_description}" in edited:
                system_prompt = edited.strip()
                break
            console.print(
                "[red]The system prompt must keep the {concept_description} placeholder.[/red]"
            )

    # 8. build + validate
    try:
        cfg = CustomClassifierConfig(
            wikibase_id=wid,
            sampling=sampling,
            concept_overrides=ConceptOverrides(
                definition=overrides.get("definition"),
                description=overrides.get("description"),
            ),
            llm=LLMClassifierConfig(
                model_name=model_name,
                system_prompt_template=system_prompt,
                labelling_guidelines=guidelines,
                related_definitions=related,
            ),
        )
    except ValidationError as e:
        console.print("[red]Could not build a valid config:[/red]")
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            console.print(f"    - {loc}: {err['msg']}")
        raise typer.Exit(code=1)

    # 9. write
    out = config_dir / f"q{wid.numeric}.yaml"
    if out.exists() and not typer.confirm(
        f"\n{out.name} exists. Overwrite?", default=False
    ):
        raise typer.Abort()
    data = cfg.model_dump(mode="json", exclude_defaults=True)
    out.write_text(
        yaml.dump(
            data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=4096
        )
    )

    # 10. re-validate the written file
    if errors := validate_file(out):
        console.print(f"[red]Written {out.name} but it failed validation:[/red]")
        for e in errors:
            console.print(f"    - {e}")
        raise typer.Exit(code=1)
    console.print(f"\n[green]Created and validated {out}[/green]")


if __name__ == "__main__":
    app()
