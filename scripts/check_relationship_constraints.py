import json
from logging import getLogger

from tqdm import tqdm

from src.wikibase import WikibaseSession

logger = getLogger(__name__)

wikibase = WikibaseSession()


def use_cache(func):
    CACHE = {}

    def wrapper(*args, **kwargs):
        if args[0] in CACHE:
            return CACHE[args[0]]
        result = func(*args, **kwargs)
        CACHE[args[0]] = result
        return result

    return wrapper


@use_cache
def get_item_claims(item_id: str):
    response = wikibase.session.get(
        url=wikibase.api_url,
        params={
            "action": "wbgetentities",
            "format": "json",
            "ids": item_id,
        },
    ).json()
    return response["entities"][item_id]["claims"]


# get a list of every item in the concept store
all_pages_response = wikibase.session.get(
    url=wikibase.api_url,
    params={
        "action": "query",
        "format": "json",
        "list": "allpages",
        "apnamespace": "120",
        # NOTE: this will work up to a limit of 5000 item pages in the concept store.
        # Beyond that, we'll need to start paginating over the results
        "aplimit": "max",
    },
).json()


missing_claims = []
for item in all_pages_response["query"]["allpages"]:
    page_id = item["title"].replace("Item:", "")
    logger.info("Checking relationships for %s", page_id)

    claims = get_item_claims(page_id)

    # if there's a P1 claim, we want to see a corresponding P2 claim on the target item
    if "P1" in claims:
        p1_items = [
            item["mainsnak"]["datavalue"]["value"]["id"] for item in claims["P1"]
        ]
        for target_item_id in p1_items:
            target_item_claims = get_item_claims(target_item_id)
            try:
                target_p2_items = [
                    item["mainsnak"]["datavalue"]["value"]["id"]
                    for item in target_item_claims["P2"]
                ]
                assert page_id in target_p2_items
            except (KeyError, AssertionError):
                missing_claims.append((page_id, target_item_id, "P2"))
                logger.error(
                    "%s has a P2 claim pointing to %s, but %s does not have a P1 claim pointing to %s",
                    page_id,
                    target_item_id,
                    target_item_id,
                    page_id,
                )

    # if there's a P2 claim, we want to see a corresponding P1 claim on the target item
    if "P2" in claims:
        p2_items = [
            item["mainsnak"]["datavalue"]["value"]["id"] for item in claims["P2"]
        ]
        for target_item_id in p2_items:
            target_item_claims = get_item_claims(target_item_id)
            try:
                target_p1_items = [
                    item["mainsnak"]["datavalue"]["value"]["id"]
                    for item in target_item_claims["P1"]
                ]
                assert page_id in target_p1_items
            except (KeyError, AssertionError):
                missing_claims.append((page_id, target_item_id, "P1"))
                logger.error(
                    "%s has a P1 claim pointing to %s, but %s does not have a P2 claim pointing to %s",
                    page_id,
                    target_item_id,
                    target_item_id,
                    page_id,
                )

    # if there's a P3 claim, we want to see a corresponding P3 claim on the target item
    if "P3" in claims:
        p3_items = [
            item["mainsnak"]["datavalue"]["value"]["id"] for item in claims["P3"]
        ]
        for target_item_id in p3_items:
            target_item_claims = get_item_claims(target_item_id)
            try:
                target_p3_items = [
                    item["mainsnak"]["datavalue"]["value"]["id"]
                    for item in target_item_claims["P3"]
                ]
                assert page_id in target_p3_items
            except (KeyError, AssertionError):
                missing_claims.append((page_id, target_item_id, "P3"))
                logger.error(
                    "%s has a P3 claim pointing to %s, but %s does not have a P3 claim pointing to %s",
                    page_id,
                    target_item_id,
                    target_item_id,
                    page_id,
                )

if missing_claims:
    logger.info("Creating missing claims")

    for page_id, target_item_id, property_id in tqdm(missing_claims):
        create_claim_response = wikibase.session.post(
            url=wikibase.api_url,
            data={
                "action": "wbcreateclaim",
                "format": "json",
                "entity": page_id,
                "property": property_id,
                "snaktype": "value",
                "value": json.dumps({"entity-type": "item", "id": target_item_id}),
                "token": wikibase.csrf_token,
                "bot": True,
                "summary": "Adding missing relationship claim",
            },
        ).json()
        logger.info(f"Created claim: {create_claim_response}")

logger.info("All relationships are consistent!")
