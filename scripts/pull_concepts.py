"""
pull_concepts.py - Script to pull concepts from wikibase and create a dictionary of concepts.

Usage:
    python3 pull_concepts.py

or in code:
    
    from pull_concepts import request_concepts, transform_and_filter

    raw_results = request_concepts() # Can raise RuntimeError
    entities = transform_and_filter(raw_results)
"""

import json
from typing import Any, Callable, Mapping, Sequence, Tuple
import requests

# Types
Json = Mapping[str, Any]

# Define the endpoint URL
WIKIBASE_SPARQL_URL = "https://climatepolicyradar.wikibase.cloud/query/sparql"

# Define the SPARQL query
SPARQL_QUERY = """
SELECT ?subject ?predicate ?object ?subjectLabel ?predicateLabel ?objectLabel WHERE {
  ?subject ?predicate ?object .
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
    ?subject rdfs:label ?subjectLabel .
    ?predicate rdfs:label ?predicateLabel .
    ?object rdfs:label ?objectLabel .
  }
}
"""


def request_concepts() -> Json:
    """Simply request the concepts from the SPARQL endpoint of the Wikibase instance.

    :raises RuntimeError: If the HTTP status code is not 200
    :return Mapping[str, Any]: The JSON response from the SPARQL query
    """
    # Define the headers for the request
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # Send the request to the SPARQL endpoint
    response = requests.post(
        WIKIBASE_SPARQL_URL, data={"query": SPARQL_QUERY}, headers=headers
    )

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(
            f"Failed to execute query. HTTP status code: {response.status_code}"
        )


def filter(entities: Json) -> Json:
    """filters the entities to only include those that are from the Wikibase instance.

    :param Mapping[str, Any] entities: entities to filter
    :return Mapping[str, Any]: entities that are from the Wikibase instance
    """
    return dict(
        (k, v)
        for k, v in entities.items()
        if v["source"].startswith("https://climatepolicyradar.wikibase.cloud/entity/Q")
    )


def _use_cpr_transform() -> Tuple[Callable[[], Json], Callable[[Json], None]]:
    """Returns the functions to transform the raw results from the Wikibase instance into a dictionary of CPR concepts."""

    relations = {}
    entities = {}

    def transform(row: Json) -> None:
        subject_value = row["subject"]["value"]
        predicate_value = row["predicate"]["value"]
        object_value = row["object"]["value"]

        subject_label = row.get("subjectLabel", {}).get("value", subject_value)
        predicate_label = row.get("predicateLabel", {}).get("value", predicate_value)
        object_label = row.get("objectLabel", {}).get("value", object_value)

        predicate = clean(predicate_label)
        obj = clean(object_label)
        subject = clean(subject_label)

        # First create new entity if not pre-existing.
        if subject not in entities:
            entities[subject] = {
                "name": subject,
                "source": subject_value,
                "altLabels": [],
                "relations": {},
            }

        # if this is a relationship then store it and record its name as the subject.
        if predicate == "directClaim":  # or predicate == "claim":
            relations[obj] = {"name": subject, "url": obj}
            return

        if predicate in relations:
            # If we have a predicate - pull the definition from the relations
            rel = relations[predicate]["name"]

            # Initialise the array if it doesn't exist
            if rel not in entities[subject]["relations"]:
                entities[subject]["relations"][rel] = []
            # Append the object as the subject's relation
            entities[subject]["relations"][rel].append(obj)
            return

        # if this is an alternative label then store it
        if predicate == "altLabel":
            entities[subject]["altLabels"].append(obj)
            return

        # if this is a descriptor then store it
        entities[subject][predicate] = obj

    def clean(text: str) -> str:
        result = text.replace("https://climatepolicyradar.wikibase.cloud/", "").replace(
            "http://schema.org/", ""
        )
        # If the string contains a hash then remove the part before the hash
        if "#" in result:
            result = result.split("#")[1]
        return result

    def get_entities() -> Json:
        return entities

    return get_entities, transform


def transform_and_filter(raw_results: Json) -> Json:
    """Transforms and filters the raw results from the Wikibase instance.

    :param Json raw_results: The raw results from the Wikibase instance
    :return Json: The transformed and filtered results
    """
    get_entities, transform = _use_cpr_transform()

    for row in raw_results["results"]["bindings"]:
        transform(row)

    entities = filter(get_entities())
    return entities


if __name__ == "__main__":
    raw_results = request_concepts()
    entities = transform_and_filter(raw_results)

    print(json.dumps(entities, indent=4))
