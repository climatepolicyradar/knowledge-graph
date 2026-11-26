from knowledge_graph.operations.wikibase_export.snapshot import Snapshot


def make_item(
    entity_id: str,
    label: str | None = None,
    claims: dict | None = None,
) -> dict:
    return {
        "type": "item",
        "id": entity_id,
        "labels": {"en": {"language": "en", "value": label}} if label else {},
        "descriptions": {},
        "aliases": {},
        "claims": claims or {},
    }


def make_snapshot(*items: dict, redirects: dict[str, str] | None = None) -> Snapshot:
    return Snapshot(
        timestamp="20261117T110000Z",
        wikibase_url="https://test.test.test",
        items=list(items),
        properties=[],
        redirects=redirects or {},
    )
