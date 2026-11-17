from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

from pydantic import BaseModel

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseSession


class Issue(BaseModel, frozen=True):
    """One problem a rule found with one entity."""

    # Filled in by run_rules from the rule's function name
    rule: str = ""
    severity: Literal["error", "warning"] = "warning"
    entity_id: str
    message: str


Rule = Callable[[Snapshot], list[Issue]]


def _item_targets(item: dict, property_id: str | None = None) -> Iterator[str]:
    """The items an item's statements point at, optionally for one property."""
    claims = item.get("claims", {})
    groups = claims.values() if property_id is None else [claims.get(property_id, [])]
    for statements in groups:
        for statement in statements:
            mainsnak = statement.get("mainsnak", {})
            # "somevalue" and "novalue" statements carry no target
            if (
                mainsnak.get("datatype") == "wikibase-item"
                and mainsnak.get("snaktype") == "value"
            ):
                yield mainsnak["datavalue"]["value"]["id"]


def missing_english_label(snapshot: Snapshot) -> list[Issue]:
    """Every item needs an English label."""
    return [
        Issue(entity_id=item["id"], message="has no English label")
        for item in snapshot.items
        if "en" not in item.get("labels", {})
    ]


def dangling_reference(snapshot: Snapshot) -> list[Issue]:
    """Statements must point at items that exist and are not redirects."""
    item_ids = {item["id"] for item in snapshot.items}
    issues = []
    for item in snapshot.items:
        for target_id in dict.fromkeys(_item_targets(item)):
            if target_id in snapshot.redirects:
                message = (
                    f"points at {target_id}, which is a redirect to "
                    f"{snapshot.redirects[target_id]}"
                )
            elif target_id not in item_ids:
                message = f"points at {target_id}, which is missing"
            else:
                continue
            issues.append(Issue(entity_id=item["id"], message=message))
    return issues


def unmirrored_subconcept(snapshot: Snapshot) -> list[Issue]:
    """A subconcept link must be recorded on both the parent and the child."""
    item_ids = {item["id"] for item in snapshot.items}

    def resolve(entity_id: str) -> str:
        return snapshot.redirects.get(entity_id, entity_id)

    # (child, parent) pairs as each side records them, following merges like
    # WikibaseSession does
    on_child = {
        (item["id"], resolve(parent_id))
        for item in snapshot.items
        for parent_id in _item_targets(item, WikibaseSession.subconcept_of_property_id)
    }
    on_parent = {
        (resolve(child_id), item["id"])
        for item in snapshot.items
        for child_id in _item_targets(item, WikibaseSession.has_subconcept_property_id)
    }

    # Targets outside the snapshot are reported by dangling_reference
    return [
        Issue(
            entity_id=child_id,
            message=(
                f"is a subconcept of {parent_id}, but {parent_id} "
                "does not list it as a subconcept"
            ),
        )
        for child_id, parent_id in sorted(on_child - on_parent)
        if parent_id in item_ids
    ] + [
        Issue(
            entity_id=parent_id,
            message=(
                f"has subconcept {child_id}, but {child_id} "
                "does not list it as a parent"
            ),
        )
        for child_id, parent_id in sorted(on_parent - on_child)
        if child_id in item_ids
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
    dangling_reference,
    unmirrored_subconcept,
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
