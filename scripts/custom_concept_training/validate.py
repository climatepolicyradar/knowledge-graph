from pathlib import Path
from typing import TYPE_CHECKING

from knowledge_graph.custom_classifier_config import CustomClassifierConfig
from knowledge_graph.exceptions import ConceptNotFoundError
from knowledge_graph.identifiers import WikibaseID

if TYPE_CHECKING:
    from knowledge_graph.wikibase import WikibaseSession

CONFIG_DIR = Path(__file__).parent / "configs"


def validate_file(path: Path) -> CustomClassifierConfig:
    """Validate one config: schema (via from_yaml) + filename matches declared wikibase_id"""
    cfg = CustomClassifierConfig.from_yaml(path)
    expected = WikibaseID(path.stem.upper())
    if cfg.wikibase_id != expected:
        raise ValueError(
            f"{path.name}: wikibase_id {cfg.wikibase_id} does not match "
            f"filename-derived {expected}"
        )
    return cfg


def validate_dir(config_dir: Path = CONFIG_DIR) -> None:
    """Validate every config in the dir."""
    for path in sorted(config_dir.glob("*.yaml")):
        validate_file(path)


def check_wikibase_ids(cfg: CustomClassifierConfig, session: "WikibaseSession") -> None:
    """Live: confirm wikibase_id + related_definitions resolve in the concept store."""
    for wid in [cfg.wikibase_id, *cfg.llm.related_definitions]:
        try:
            session.get_concept(wid)
        except ConceptNotFoundError as e:
            raise ValueError(f"{wid} does not exist in Wikibase") from e
