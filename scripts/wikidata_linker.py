import logging
from collections import Counter

import pandas as pd
from pydantic import BaseModel, computed_field
from rich.logging import RichHandler

from knowledge_graph.concept import Concept
from knowledge_graph.wikibase import WikibaseSession
from knowledge_graph.wikidata import WikidataSession

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)

logger.info("Connecting to Wikibase")
wikibase = WikibaseSession()
logger.info("Connected to Wikibase")

logger.info("Connecting to Wikidata")
wikidata = WikidataSession()
logger.info("Connected to Wikidata")


class MatchedConcept(BaseModel):  # noqa: D101
    wikibase_concept: Concept
    wikidata_concept: Concept

    def __str__(self) -> str:  # noqa: D105
        return f"{self.wikibase_concept} -> {self.wikidata_concept}"

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    @computed_field
    @property
    def wikibase_preferred_label(self) -> str:  # noqa: D102
        return self.wikibase_concept.preferred_label

    @computed_field
    @property
    def wikidata_preferred_label(self) -> str:  # noqa: D102
        return self.wikidata_concept.preferred_label

    @computed_field
    @property
    def wikibase_url(self) -> str:  # noqa: D102
        return f"{wikibase.base_url}/wiki/Item:{self.wikibase_concept.wikibase_id}"

    @computed_field
    @property
    def wikidata_url(self) -> str:  # noqa: D102
        return f"{wikidata.base_url}/wiki/{self.wikidata_concept.wikibase_id}"


wikibase_ids = wikibase.get_all_concept_ids()[450:]
matches: list[MatchedConcept] = []
for i, wikibase_id in enumerate(wikibase_ids):
    logger.info(f"Processing concept #{i + 1} of {len(wikibase_ids)}")
    logger.info(f"Getting concept from Wikibase: {wikibase_id}")
    concept = wikibase.get_concept(wikibase_id)

    logger.info(
        f'Searching for the same concept in Wikidata: "{concept.preferred_label}"'
    )

    if wikidata_search_results := wikidata.search_concepts(
        concept.preferred_label, limit=1
    ):
        result = wikidata_search_results[0]
    else:
        logger.info(f'No result found in Wikidata for "{concept.preferred_label}"')
        results: list[Concept] = []
        for label in concept.all_labels:
            logger.info(f'Searching for alternative label: "{label}"')
            wikidata_search_results = wikidata.search_concepts(label, limit=3)
            results.extend(wikidata_search_results)
        try:
            result, count = Counter(results).most_common(1)[0]
        except IndexError:
            logger.info(f'No result found in Wikidata for any labels of "{concept}"')
            result = None
            count = 0

    if result:
        match = MatchedConcept(wikibase_concept=concept, wikidata_concept=result)
        matches.append(match)
        logger.info(f"There may be a link: {match!r}")
    else:
        logger.info(f'No link found between "{concept}" and Wikidata')

    if i + 1 % 25 == 0 or i + 1 == len(wikibase_ids):
        df = pd.DataFrame(
            [
                match.model_dump(exclude={"wikibase_concept", "wikidata_concept"})
                for match in matches
            ]
        )
        df.to_csv("wikidata_matches.csv", index=False)
