from knowledge_graph.operations.wikibase_export.snapshot import Snapshot


def item_claim(property_id: str, *target_ids: str) -> dict:
    """Statements linking an item to other items, as Wikibase returns them."""
    return {
        property_id: [
            {
                "mainsnak": {
                    "snaktype": "value",
                    "property": property_id,
                    "datatype": "wikibase-item",
                    "datavalue": {
                        "value": {"entity-type": "item", "id": target_id},
                        "type": "wikibase-entityid",
                    },
                },
                "type": "statement",
                "rank": "normal",
            }
            for target_id in target_ids
        ]
    }


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
