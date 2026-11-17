from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot
from knowledge_graph.utils import get_logger


class Issue(BaseModel, frozen=True):
    """One problem a rule found with one entity."""

    # Filled in by run_rules from the rule's function name
    rule: str = ""
    severity: Literal["error", "warning"] = "warning"
    entity_id: str
    message: str


Rule = Callable[[Snapshot], list[Issue]]


def missing_english_label(snapshot: Snapshot) -> list[Issue]:
    """Every item needs an English label."""
    return [
        Issue(entity_id=item["id"], message="has no English label")
        for item in snapshot.items
        if "en" not in item.get("labels", {})
    ]


def duplicate_english_label(snapshot: Snapshot) -> list[Issue]:
    """No two items should share an English label."""
    ids_by_label: dict[str, list[str]] = defaultdict(list)
    for item in snapshot.items:
        if label := item.get("labels", {}).get("en", {}).get("value"):
            ids_by_label[label.strip().casefold()].append(item["id"])

    issues = []
    for ids in ids_by_label.values():
        if len(ids) < 2:
            continue
        for item_id in ids:
            others = ", ".join(other for other in ids if other != item_id)
            issues.append(
                Issue(
                    entity_id=item_id,
                    message=f"shares its English label with {others}",
                )
            )
    return issues


RULES: list[Rule] = [
    missing_english_label,
    duplicate_english_label,
]


def run_rules(
    snapshot: Snapshot, rules: Sequence[Rule] = RULES
) -> tuple[list[Issue], list[str]]:
    """Run every rule, returning the issues found and the rules that raised."""
    logger = get_logger()
    issues: list[Issue] = []
    failed_rules: list[str] = []
    for rule in rules:
        try:
            issues.extend(
                issue.model_copy(update={"rule": rule.__name__})
                for issue in rule(snapshot)
            )
        except Exception:
            logger.exception("Rule %s raised", rule.__name__)
            failed_rules.append(rule.__name__)
    return issues, failed_rules
