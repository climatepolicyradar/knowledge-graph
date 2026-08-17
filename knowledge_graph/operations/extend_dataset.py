"""
Extend-dataset operation: reusable, Prefect-free domain logic.

Adds more labelled passages to a dataset which already exists in Argilla,
deduplicating the input passages against the text already in the dataset so that
annotators never see the same passage twice.
"""

from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import Identifier, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.utils import get_logger


class NotEnoughNewPassagesError(Exception):
    """Raised when deduplication leaves fewer new passages than the caller requires."""

    def __init__(self, wikibase_id: WikibaseID | str, requested: int, available: int):
        self.message = (
            f"Only {available} of the input passages for {wikibase_id} are new, but "
            f"{requested} were requested. Either the input sample was too small — sample "
            f"more passages — or the candidate pool for {wikibase_id} is exhausted, in "
            f'which case switch to dataset_name="combined" for a much larger pool, '
            f"rather than raising sample_size. Otherwise lower the limit."
        )
        super().__init__(self.message)


def get_passage_ids_in_argilla(
    wikibase_id: WikibaseID,
    workspace: str = "knowledge-graph",
    argilla_api_url: str | None = None,
    argilla_api_key: str | None = None,
) -> set[str]:
    """Return the IDs of the passages already in a concept's Argilla dataset."""
    logger = get_logger()

    argilla = ArgillaSession(api_url=argilla_api_url, api_key=argilla_api_key)
    dataset = argilla.get_dataset(wikibase_id, workspace=workspace)
    passage_ids: set[str] = {
        Identifier.generate(record.fields.get("text", "")) for record in dataset.records
    }
    logger.info(
        f"✅ Found {len(passage_ids)} passages already in dataset '{dataset.name}'"
    )
    return passage_ids


def run_extend_dataset(
    wikibase_id: WikibaseID,
    labelled_passages: list[LabelledPassage],
    workspace: str = "knowledge-graph",
    limit: int | None = 130,
    suggestion_model=None,
    require_full_limit: bool = False,
    existing_passage_ids: set[str] | None = None,
    argilla_api_url: str | None = None,
    argilla_api_key: str | None = None,
) -> list[LabelledPassage]:
    """
    Add more labelled passages to a dataset which already exists in Argilla.

    Deduplicates the input passages by the passage text's Identifier against what's already in
    Argilla, before applying the limit — so `limit` is the number of new passages added,
    not the number of inputs considered.
    """
    logger = get_logger()

    argilla = ArgillaSession(api_url=argilla_api_url, api_key=argilla_api_key)
    logger.info("✅ Connected to Argilla")

    # Get existing dataset from Argilla
    logger.info(f"Looking for existing dataset for {wikibase_id}")
    dataset = argilla.get_dataset(wikibase_id, workspace=workspace)
    if existing_passage_ids is None:
        existing_passage_ids = {
            Identifier.generate(record.fields.get("text", ""))
            for record in dataset.records
        }
    logger.info(
        f"✅ Found existing dataset '{dataset.name}' with "
        f"{len(existing_passage_ids)} distinct passages"
    )

    # Deduplicate input passages against Argilla
    lp_length_before = len(labelled_passages)
    labelled_passages = [
        lp
        for lp in labelled_passages
        if Identifier.generate(lp.text) not in existing_passage_ids
    ]
    logger.info(
        f"{len(labelled_passages)}/{lp_length_before} input passages remaining after "
        f"deduplication"
    )

    if limit is not None and len(labelled_passages) < limit:
        if require_full_limit:
            raise NotEnoughNewPassagesError(
                wikibase_id, requested=limit, available=len(labelled_passages)
            )
        logger.warning(
            f"Only {len(labelled_passages)} new passages available, fewer than the "
            f"{limit} requested. Re-run to top up — passages added now are excluded from "
            f"the next sample."
        )
    elif limit is not None:
        logger.info(f"Limiting number of labelled passages to {limit}")
        labelled_passages = labelled_passages[:limit]

    if not labelled_passages:
        logger.warning(
            f"No new passages to add to dataset '{dataset.name}', doing nothing"
        )
        return []

    logger.info(f"Adding {len(labelled_passages)} passages to dataset")
    argilla.add_labelled_passages(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        workspace=workspace,
        suggestion_model=suggestion_model,
    )

    logger.info(
        f"✅ Successfully added {len(labelled_passages)} passages to dataset "
        f"'{dataset.name}'"
    )
    logger.info(
        f"Dataset now contains {len(existing_passage_ids) + len(labelled_passages)} "
        f"distinct passages"
    )
    return labelled_passages


def extend_dataset_locally(
    wikibase_id: WikibaseID,
    workspace: str = "knowledge-graph",
    limit: int | None = 130,
    require_full_limit: bool = False,
) -> list[LabelledPassage]:
    """
    Extend an existing Argilla dataset from the local sample file.

    Local counterpart to `run_extend_dataset`, reading the JSONL that the sample
    operation writes to `data/processed/sampled_passages/` instead of taking passages
    from a W&B artifact.
    """
    logger = get_logger()

    sampled_passages_path = (
        processed_data_dir / "sampled_passages" / f"{wikibase_id}.jsonl"
    )
    if not sampled_passages_path.exists():
        raise FileNotFoundError(
            f"Sampled passages not found for {wikibase_id}. Run the sample script "
            f"(scripts/sample.py) first."
        )

    logger.info(f"Loading sampled passages for {wikibase_id}")
    with open(sampled_passages_path, "r", encoding="utf-8") as f:
        labelled_passages = [LabelledPassage.model_validate_json(line) for line in f]
    n_annotations = sum(len(entry.spans) for entry in labelled_passages)
    logger.info(
        f"Loaded {len(labelled_passages)} labelled passages with {n_annotations} "
        f"individual annotations"
    )

    return run_extend_dataset(
        wikibase_id=wikibase_id,
        labelled_passages=labelled_passages,
        workspace=workspace,
        limit=limit,
        require_full_limit=require_full_limit,
    )
