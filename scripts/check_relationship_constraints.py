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
all_item_ids = wikibase.get_all_item_ids()


# define the constraints we want to enforce
relationship_constraints = [
    # (property_id, target_property_id)
    (wikibase.has_subconcept_property_id, wikibase.subconcept_of_property_id),
    (wikibase.subconcept_of_property_id, wikibase.has_subconcept_property_id),
    (wikibase.related_concept_property_id, wikibase.related_concept_property_id),
]

missing_claims = []
for item_id in all_item_ids:
    logger.info("Checking relationships for %s", item_id)

    claims = get_item_claims(item_id)

    for property_id, target_property_id in relationship_constraints:
        if property_id in claims:
            items = [
                item["mainsnak"]["datavalue"]["value"]["id"]
                for item in claims[property_id]
            ]
            for target_item_id in items:
                target_item_claims = get_item_claims(target_item_id)
                try:
                    target_items = [
                        item["mainsnak"]["datavalue"]["value"]["id"]
                        for item in target_item_claims[target_property_id]
                    ]
                    assert item_id in target_items
                except (KeyError, AssertionError):
                    missing_claims.append((item_id, target_item_id, target_property_id))
                    logger.error(
                        "%s has a %s claim pointing to %s, but %s does not have a %s claim pointing to %s",
                        item_id,
                        property_id,
                        target_item_id,
                        target_item_id,
                        target_property_id,
                        item_id,
                    )

if missing_claims:
    logger.info("Creating missing claims")
    for item_id, target_item_id, property_id in tqdm(missing_claims):
        create_claim_response = wikibase.add_statement(
            subject_id=item_id,
            predicate_id=property_id,
            object_id=target_item_id,
            summary="Adding missing relationship claim",
        )
        logger.info(f"Created claim: {create_claim_response}")

logger.info("All relationships are consistent!")
