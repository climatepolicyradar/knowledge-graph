from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

from pydantic import BaseModel

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseSession

logger = get_logger(__name__)


class Issue(BaseModel, frozen=True):
    """One problem a rule found with one entity."""

    rule: str
    severity: Literal["error", "warning"]
    entity_id: str
    message: str


Rule = Callable[[Snapshot], list[Issue]]


def _item_targets(item: dict, property_id: str | None = None) -> Iterator[str]:
    """The items an item's statements point at, optionally for one property."""
    for claim_property_id, statements in item.get("claims", {}).items():
        if property_id is not None and claim_property_id != property_id:
            continue
        for statement in statements:
            mainsnak = statement.get("mainsnak", {})
            if mainsnak.get("datatype") != "wikibase-item":
                continue
            # "somevalue" and "novalue" statements carry no target
            if mainsnak.get("snaktype") != "value":
                continue
            yield mainsnak["datavalue"]["value"]["id"]


def missing_english_label(snapshot: Snapshot) -> list[Issue]:
    """Every item needs an English label."""
    return [
        Issue(
            rule="missing_english_label",
            severity="warning",
            entity_id=item["id"],
            message="has no English label",
        )
        for item in snapshot.items
        if "en" not in item.get("labels", {})
    ]


def dangling_reference(snapshot: Snapshot) -> list[Issue]:
    """Statements must point at items that exist and are not redirects."""
    item_ids = snapshot.item_ids()
    issues = []
    for item in snapshot.items:
        for target_id in _item_targets(item):
            if target_id in snapshot.redirects:
                message = (
                    f"points at {target_id}, which is a redirect to "
                    f"{snapshot.redirects[target_id]}"
                )
            elif target_id not in item_ids:
                message = f"points at {target_id}, which is missing"
            else:
                continue
            issues.append(
                Issue(
                    rule="dangling_reference",
                    severity="warning",
                    entity_id=item["id"],
                    message=message,
                )
            )
    return issues


def unmirrored_subconcept(snapshot: Snapshot) -> list[Issue]:
    """A subconcept link must be recorded on both the parent and the child."""
    subconcept_of = WikibaseSession.subconcept_of_property_id
    has_subconcept = WikibaseSession.has_subconcept_property_id
    parents = {
        item["id"]: set(_item_targets(item, subconcept_of)) for item in snapshot.items
    }
    children = {
        item["id"]: set(_item_targets(item, has_subconcept)) for item in snapshot.items
    }

    issues = []
    for item in snapshot.items:
        item_id = item["id"]
        # Targets outside the snapshot are reported by dangling_reference
        for parent_id in sorted(parents[item_id] & children.keys()):
            if item_id not in children[parent_id]:
                issues.append(
                    Issue(
                        rule="unmirrored_subconcept",
                        severity="warning",
                        entity_id=item_id,
                        message=(
                            f"is a subconcept of {parent_id}, but {parent_id} "
                            "does not list it as a subconcept"
                        ),
                    )
                )
        for child_id in sorted(children[item_id] & parents.keys()):
            if item_id not in parents[child_id]:
                issues.append(
                    Issue(
                        rule="unmirrored_subconcept",
                        severity="warning",
                        entity_id=item_id,
                        message=(
                            f"has subconcept {child_id}, but {child_id} "
                            "does not list it as a parent"
                        ),
                    )
                )
    return issues


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
                    rule="duplicate_english_label",
                    severity="warning",
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
    issues: list[Issue] = []
    failed_rules: list[str] = []
    for rule in rules:
        try:
            issues.extend(rule(snapshot))
        except Exception:
            # One broken rule must not stop the export or the other rules
            logger.exception("Rule %s raised", rule.__name__)
            failed_rules.append(rule.__name__)
    return issues, failed_rules
