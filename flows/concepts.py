from prefect import flow

from flows.utils import get_logger
from knowledge_graph.concept import Concept


@flow(log_prints=True)
def get_concept_class_fields():
    logger = get_logger()
    concept_fields = Concept.__fields__.keys()
    logger.info(f"Concept class fields: {concept_fields}")


if __name__ == "__main__":
    get_concept_class_fields()
