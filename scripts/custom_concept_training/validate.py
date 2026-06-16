from pathlib import Path

from knowledge_graph.custom_classifier_config import CustomClassifierConfig
from knowledge_graph.identifiers import WikibaseID

CONFIG_DIR = Path(__file__).parent / "configs"


def expected_wikibase_id(path: Path) -> WikibaseID:
    """Derive the WikibaseID a file must declare from its name (q32.yaml -> Q32)."""
    return WikibaseID(path.stem.upper())


def validate_file(path: Path) -> list[str]:
    """Offline: schema + filename matches declared wikibase_id."""
    try:
        cfg = CustomClassifierConfig.from_yaml(path)
    except Exception as e:  # pydantic ValidationError, malformed YAML, IO, etc.
        return [str(e)]

    errors: list[str] = []
    try:
        expected = expected_wikibase_id(path)
    except ValueError:
        return [f"filename {path.name!r} is not of the form q<number>.yaml"]
    if cfg.wikibase_id != expected:
        errors.append(
            f"wikibase_id {cfg.wikibase_id} does not match filename-derived {expected}"
        )
    return errors


def validate_dir(config_dir: Path = CONFIG_DIR) -> dict[Path, list[str]]:
    """Run validate_file on every YAML plus a cross-file duplicate-concept check."""
    paths = sorted(config_dir.glob("*.yaml"))
    results: dict[Path, list[str]] = {p: validate_file(p) for p in paths}

    seen: dict[WikibaseID, Path] = {}
    for p in paths:
        try:
            wid = expected_wikibase_id(p)
        except ValueError:
            continue
        if wid in seen:
            results[p].append(
                f"duplicate concept {wid}; already defined in {seen[wid].name}"
            )
        else:
            seen[wid] = p
    return results


def check_wikibase_ids(cfg: CustomClassifierConfig, session) -> list[str]:
    """Live: confirm wikibase_id + related_definitions resolve in the concept store."""
    from knowledge_graph.wikibase import ConceptNotFoundError

    errors: list[str] = []
    for wid in [cfg.wikibase_id, *cfg.llm.related_definitions]:
        try:
            session.get_concept(wid)
        except ConceptNotFoundError:
            errors.append(f"{wid} does not exist in Wikibase")
    return errors
