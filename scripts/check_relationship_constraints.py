from tqdm import tqdm

from src.wikibase import WikibaseSession

wikibase = WikibaseSession()

CACHE = {}


def get_item_claims(item_id: str):
    if item_id in CACHE:
        claims = CACHE[item_id]
    else:
        response = wikibase.session.get(
            url=wikibase.api_url,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": item_id,
            },
        ).json()
        claims = response["entities"][item_id]["claims"]
        CACHE[item_id] = claims
    return claims


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

progress_bar = tqdm(all_pages_response["query"]["allpages"])

for item in progress_bar:
    page_id = item["title"].replace("Item:", "")
    progress_bar.set_description(f"Checking claims for {page_id}")

    claims = get_item_claims(page_id)

    if "P1" in claims:
        # if there's a P1 claim, we want to see a corresponding P2 claim on the target item
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
            except KeyError:
                raise AssertionError(f"No P2 claims found on {target_item_id}")
            except AssertionError:
                raise AssertionError(
                    f"{page_id} has a P1 claim pointing to {target_item_id}, but "
                    f"{target_item_id} does not have a P2 claim pointing to {page_id} "
                )

    if "P2" in claims:
        # if there's a P2 claim, we want to see a corresponding P1 claim on the target item
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
            except KeyError:
                raise AssertionError(f"No P1 claims found on {target_item_id}")
            except AssertionError:
                raise AssertionError(
                    f"{page_id} has a P2 claim pointing to {target_item_id}, but "
                    f"{target_item_id} does not have a P1 claim pointing to {page_id} "
                )

    if "P3" in claims:
        # if there's a P3 claim, we want to see a corresponding P3 claim on the target item
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
            except KeyError:
                raise AssertionError(f"No P3 claims found on {target_item_id}")
            except AssertionError:
                raise AssertionError(
                    f"{page_id} has a P3 claim pointing to {target_item_id}, but "
                    f"{target_item_id} does not have a P3 claim pointing to {page_id} "
                )
