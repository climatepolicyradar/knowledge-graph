import asyncio
import base64
import contextlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Awaitable
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Generator, Optional, Union

from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.deployments import run_deployment
from prefect.logging import get_logger, get_run_logger

from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.span import Span

DEFAULT_BATCH_SIZE = 25
# There's a limit to the size (512kb [1]) to flow run parameters. That
# means that we need to limit the number of concepts in a partial
# update.
#
# We've calculated this value [2] based on representative data. It may
# not be perfect, since, the actual values vary, and thus increase or
# decrease the serialised size. It may need to be tweaked.
#
# [1] https://docs.prefect.io/v3/develop/write-flows
# [2]
# >>> params = {"document_concepts": [["227", "{\"id\":\"Q368\",\"name\":\"marine risk\",\"parent_concepts\":[{\"id\":\"Q949\",\"name\":\"\"}],\"parent_concept_ids_flat\":\"Q949,\",\"model\":\"KeywordClassifier(\\\"marine risk\\\")\",\"end\":123,\"start\":106,\"timestamp\":\"2025-01-09T10:29:25.598270\"}"]]*1500, "document_import_id": "CCLW.executive.10272.4889"}
# >>> round(len(bytes(json.dumps(params).encode("utf-8")))/1024)
# 397
MAX_CONCEPTS_IN_PARTIAL_UPDATE = 1500


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: Optional[str] = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    bucket_region: str = "eu-west-1"
    # An instance of VespaSearchAdapter.
    #
    # E.g.
    #
    # VespaSearchAdapter(
    #   instance_url="https://vespa-instance-url.com",
    #   cert_directory="certs/"
    # )
    vespa_search_adapter: Optional[VespaSearchAdapter] = None
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])
    as_subflow: bool = True

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        logger = get_run_logger()

        config = cls()

        if not config.cache_bucket:
            logger.info(
                "no cache bucket provided, getting it from Prefect job variable"
            )
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        return config


def get_vespa_search_adapter_from_aws_secrets(
    cert_dir: str,
    vespa_instance_url_param_name: str = "VESPA_INSTANCE_URL",
    vespa_public_cert_param_name: str = "VESPA_PUBLIC_CERT_READ",
    vespa_private_key_param_name: str = "VESPA_PRIVATE_KEY_READ",
) -> VespaSearchAdapter:
    """
    Get a VespaSearchAdapter instance by retrieving secrets from AWS Secrets Manager.

    We then save the secrets to local files in the cert_dir directory and instantiate
    the VespaSearchAdapter.
    """
    cert_dir_path = Path(cert_dir)
    if not cert_dir_path.exists():
        raise FileNotFoundError(f"Certificate directory does not exist: {cert_dir}")

    vespa_instance_url = get_aws_ssm_param(vespa_instance_url_param_name)
    vespa_public_cert_encoded = get_aws_ssm_param(vespa_public_cert_param_name)
    vespa_private_key_encoded = get_aws_ssm_param(vespa_private_key_param_name)

    vespa_public_cert = base64.b64decode(vespa_public_cert_encoded).decode("utf-8")
    vespa_private_key = base64.b64decode(vespa_private_key_encoded).decode("utf-8")

    cert_path = cert_dir_path / "cert.pem"
    key_path = cert_dir_path / "key.pem"

    with open(cert_path, "w", encoding="utf-8") as f:
        f.write(vespa_public_cert)

    with open(key_path, "w", encoding="utf-8") as f:
        f.write(vespa_private_key)

    return VespaSearchAdapter(
        instance_url=vespa_instance_url,
        cert_directory=str(cert_dir_path),
    )


def s3_obj_generator_from_s3_prefixes(
    s3_prefixes: list[str],
) -> Generator[
    tuple[
        # Object: key
        str,
        # Object: body. It's a list since we store the labelled
        # passages as JSONL.
        list[dict],
    ],
    None,
    None,
]:
    """Return a generator that yields objects from a list of S3 prefixes."""
    logger = get_logger()
    for s3_prefix in s3_prefixes:
        try:
            object_keys = _get_s3_keys_with_prefix(s3_prefix=s3_prefix)
            bucket = Path(s3_prefix).parts[1]
            for key in object_keys:
                obj = _s3_object_read_text(s3_path=(os.path.join("s3://", bucket, key)))
                yield key, json.loads(obj)
        except Exception as e:
            logger.error(
                "failed to yield object from S3 prefix",
                extra={"error": str(e)},
            )
            continue


def s3_obj_generator_from_s3_paths(
    s3_paths: list[str],
) -> Generator[
    tuple[
        # Object: key
        str,
        # Object: body. It's a list since we store the labelled
        # passages as JSONL.
        list[dict],
    ],
    None,
    None,
]:
    """
    Return a generator that yields objects from a list of S3 paths.

    We extract the key from the S3 path by removing the first two
    elements in the path.

    E.g. "s3://bucket/prefix/file.json" -> "prefix/file.json"
    """
    logger = get_logger()
    for s3_path in s3_paths:
        try:
            key = "/".join(Path(s3_path).parts[2:])
            body = json.loads(_s3_object_read_text(s3_path=s3_path))
            yield key, body
        except Exception as e:
            logger.error(
                "failed to yield object from S3 path",
                extra={"error": str(e)},
            )
            continue


def labelled_passages_generator(
    generator_func: Generator,
) -> Generator[
    tuple[
        # Object: key
        str,
        # Object: body. It's a list since we store the labelled
        # passages as JSONL.
        list[LabelledPassage],
    ],
    None,
    None,
]:
    """
    Transforms the S3 objects bodies into LabelledPassages objects.

    Effectively a wrapper for the other generator. Each yielded object
    from the generator to a LabelledPassage object.
    """
    for s3_key, obj in generator_func:
        yield s3_key, [LabelledPassage(**labelled_passage) for labelled_passage in obj]


def get_document_passages_from_vespa(
    document_import_id: str, vespa_search_adapter: VespaSearchAdapter
) -> list[tuple[str, VespaPassage]]:
    """
    Retrieve all the passages for a document in Vespa.

    params:
    - document_import_id: The document import id for a unique family document.
    """
    logger = get_logger()

    logger.info(f"Getting document passages from Vespa: {document_import_id}")

    vespa_query_response = vespa_search_adapter.client.query(
        yql=(
            # trunk-ignore(bandit/B608)
            "select * from document_passage where family_document_ref contains "
            f'"id:doc_search:family_document::{document_import_id}"'
        )
    )
    logger.info(
        f"Vespa search response for document: {document_import_id} "
        f"with {len(vespa_query_response.hits)} hits"
    )

    return [
        (passage["id"], VespaPassage.model_validate(passage["fields"]))
        for passage in vespa_query_response.hits
    ]


def get_text_block_id_from_labelled_passage(labelled_passage: LabelledPassage) -> str:
    """Identify the text block id that a labelled passage relates to."""
    return labelled_passage.id


def get_model_from_span(span: Span) -> str:
    """
    Get the model used to label the span.

    Labellers are stored in a list, these can contain many labellers as seen in the
    example below, referring to human and machine annotators.

    [
        "alice",
        "bob",
        "68edec6f-fe74-413d-9cf1-39b1c3dad2c0",
        'KeywordClassifier("extreme weather")',
    ]

    In the context of inference the labellers array should only hold the model used to
    label the span as seen in the example below.

    [
        'KeywordClassifier("extreme weather")',
    ]
    """
    if len(span.labellers) != 1:
        raise ValueError(
            f"Span has more than one labeller. Expected 1, got {len(span.labellers)}."
        )
    return span.labellers[0]


def get_passage_for_text_block(
    text_block_id: str, document_passages: list[tuple[str, VespaPassage]]
) -> Union[tuple[str, str, VespaPassage], tuple[None, None, None]]:
    """
    Return the data ID, passage and passage ID that a text block relates to.

    Concepts relate to a specific passage or text block within a document
    and therefore, we must find the relevant text block to update when running
    partial updates.

    Extract data ID (last element after "::"), e.g., "CCLW.executive.10014.4470.623"
    from passage_id like "id:doc_search:document_passage::CCLW.executive.10014.4470.623".
    """
    passage = next(
        (
            passage
            for passage in document_passages
            if passage[1].text_block_id == text_block_id
        ),
        None,
    )

    if passage:
        data_id = passage[0].split("::")[-1]
        passage_id = passage[0]
        passage_content = passage[1]
        return data_id, passage_id, passage_content

    return None, None, None


def get_parent_concepts_from_concept(
    concept: Concept,
) -> tuple[list[dict], str]:
    """
    Extract parent concepts from a Concept object.

    Currently we pull the name from the Classifier used to label the passage, this
    doesn't hold the concept id. This is a temporary solution that is not desirable as
    the relationship between concepts can change frequently and thus shouldn't be
    coupled with inference.
    """
    parent_concepts = [
        {"id": subconcept, "name": ""} for subconcept in concept.subconcept_of
    ]
    parent_concept_ids_flat = (
        ",".join([parent_concept["id"] for parent_concept in parent_concepts]) + ","
    )

    return parent_concepts, parent_concept_ids_flat


def convert_labelled_passages_to_concepts(
    labelled_passages: list[LabelledPassage],
) -> list[
    tuple[
        # Text block (aka span) ID
        str,
        VespaConcept,
    ]
]:
    """
    Convert a labelled passage to a list of VespaConcept objects and their text block ID.

    The labelled passage contains a list of spans relating to concepts
    that we must convert to VespaConcept objects.
    """
    logger = get_run_logger()

    concepts = []

    for labelled_passage in labelled_passages:
        # The concept used to label the passage holds some information on the parent
        # concepts and thus this is being used as a temporary solution for providing
        # the relationship between concepts. This has the downside that it ties a
        # labelled passage to a particular concept when in fact the Spans that a
        # labelled passage has can be labelled by multiple concepts.
        concept = Concept.model_validate(labelled_passage.metadata["concept"])
        parent_concepts, parent_concept_ids_flat = get_parent_concepts_from_concept(
            concept=concept
        )
        text_block_id = get_text_block_id_from_labelled_passage(labelled_passage)

        # This expands the list from `n` for `LabelledPassages` to `n` for `Spans`
        for span_idx, span in enumerate(labelled_passage.spans):
            if span.concept_id is None:
                # Include the Span index since Span's don't have IDs
                logger.error(
                    f"span concept ID is missing: LabelledPassage.id={labelled_passage.id}, Span index={span_idx}"
                )
                continue

            if not span.timestamps:
                logger.error(
                    f"span timestamps are missing: LabelledPassage.id={labelled_passage.id}, Span index={span_idx}"
                )
                continue

            concepts.append(
                (
                    text_block_id,
                    VespaConcept(
                        id=span.concept_id,
                        name=concept.preferred_label,
                        parent_concepts=parent_concepts,
                        parent_concept_ids_flat=parent_concept_ids_flat,
                        model=get_model_from_span(span),
                        end=span.end_index,
                        start=span.start_index,
                        # these timestamps _should_ all be the same,
                        # but just in case, take the latest
                        timestamp=max(span.timestamps),
                    ),
                )
            )

    return concepts


def get_updated_passage_concepts(
    passage: VespaPassage, concepts: list[VespaConcept]
) -> list[dict]:
    """
    Update a passage's concepts with the new concept.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in vespa for a
    model is not useful.

    It is also, not possible to duplicate a Concept object in the concepts array as we
    are removing all instances where the model is the same.
    """
    if not passage.concepts:
        return [concept.model_dump(mode="json") for concept in concepts]

    new_concept_models = {concept.model for concept in concepts}

    existing_concepts_to_keep = [
        concept
        for concept in passage.concepts
        if concept.model not in new_concept_models
    ]

    updated_concepts = existing_concepts_to_keep + concepts

    return [concept_.model_dump(mode="json") for concept_ in updated_concepts]


def group_concepts_on_text_block(
    document_concepts: list[tuple[str, VespaConcept]],
) -> dict[str, list[VespaConcept]]:
    """
    Group concepts on text block ID.

    Concepts relate to a specific passage or text block within a document and therefore,
    we must group the concept updates to all of them at once.
    """
    concepts_grouped = {}
    for text_block_id, concept in document_concepts:
        if text_block_id in concepts_grouped:
            concepts_grouped[text_block_id].append(concept)
        else:
            concepts_grouped[text_block_id] = [concept]
    return concepts_grouped


def get_concepts_counts(
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            VespaConcept,
        ]
    ],
) -> dict[
    # Concept
    str,
    # Count
    int,
]:
    """
    Get the concepts' counts for a document.

    The concept counts are used to update the family document with the counts of the
    concepts related to the document.
    """
    concept_counts = defaultdict(int)
    for _, concept in document_concepts:
        concept_counts[concept.id] += 1
    return dict(concept_counts)


@flow
async def run_partial_updates_of_concepts_for_document_passages(
    document_import_id: str,
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            Union[
                VespaConcept,
                # Serialised JSON of object
                str,
            ],
        ]
    ],
    vespa_search_adapter: Optional[VespaSearchAdapter] = None,
) -> None:
    """
    Run partial update for VespaConcepts on text blocks for a document.

    This is done in the document_passage index.

    Assumptions:

    - The ID field of the VespaConcept object holds the
    context of the text block that it relates to. E.g. the concept ID
    1.10 would relate to the text block ID 10.
    """
    logger = get_run_logger()

    # We want the directory used for the `VespaSearchAdapter` to be
    # automatically cleaned up.
    #
    # To do this, we rely on the `tempfile.TemporaryDirectory`'s behaviour,
    # or, a `contextlib.nullcontext` no-op, if a temporary directory
    # wasn't needed.
    if vespa_search_adapter is None:
        logger.info("no Vespa search adapter, getting it from AWS secrets")
        cm = tempfile.TemporaryDirectory()

        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=cm.name,  # type: ignore
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )
    else:
        logger.info("Vespa search adapter provided")
        cm = contextlib.nullcontext()

    with cm:
        logger.info(
            "getting document passages from Vespa for document "
            f"import ID {document_import_id}"
        )
        document_passages = get_document_passages_from_vespa(
            document_import_id=document_import_id,
            vespa_search_adapter=vespa_search_adapter,
        )

        if not document_passages:
            logger.error(
                f"No hits for document import ID {document_import_id} in Vespa. "
                "Either the document doesn't exist or there are no passages related to "
                "the document.",
            )
            raise ValueError(
                f"No passages found for document in Vespa: {document_import_id}"
            )

        loaded_document_concepts = maybe_load_document_concepts(document_concepts)
        grouped_concepts = group_concepts_on_text_block(loaded_document_concepts)

        logger.info(
            f"starting partial updates for {len(grouped_concepts)} grouped concepts"
        )

        batches = iterate_batch(list(grouped_concepts.items()))

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id,
                    document_passages,
                    vespa_search_adapter,
                    concepts,
                )
                for text_block_id, concepts in batch
            ]

            logger.info(f"gathering partial updates tasks for batch {batch_num}")
            results = await asyncio.gather(
                *partial_update_tasks, return_exceptions=True
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Get concept
                    concept = batch[i][0]

                    logger.error(
                        f"failed to do partial update for concept `{concept}`: {str(result)}",
                    )

        # Convert document_concepts to concept counts
        concept_counts = get_concepts_counts(loaded_document_concepts)

        # Run partial updates of the family documents to add the concept counts
        if not concept_counts:
            vespa_search_adapter.client.update_data(  # pyright: ignore[reportOptionalMemberAccess]
                schema="family_document",
                namespace="doc_search",
                data_id=document_import_id,
                fields={
                    "concept_counts": {
                        concept_id: count
                        for concept_id, count in concept_counts.items()
                    }
                },
            )

            logger.info(
                f"updated concept metadata for family_document: {document_import_id}",
            )


async def partial_update_text_block(
    text_block_id: str,
    document_passages: list[tuple[str, VespaPassage]],
    vespa_search_adapter: VespaSearchAdapter,
    concepts: list[VespaConcept],
) -> None:
    """Partial update a singular text block and its concepts."""
    logger = get_run_logger()

    data_id, passage_id, passage_for_text_block = get_passage_for_text_block(
        text_block_id, document_passages
    )

    if data_id and passage_id and passage_for_text_block:
        vespa_search_adapter.client.update_data(  # pyright: ignore[reportOptionalMemberAccess]
            schema="document_passage",
            namespace="doc_search",
            data_id=data_id,
            fields={
                "concepts": get_updated_passage_concepts(
                    passage_for_text_block, concepts
                )
            },
        )
    else:
        logger.error(f"No passages found for text block: {text_block_id}")


async def convert_labelled_passages_to_document_concepts(
    document_labelled_passages: list[
        tuple[
            # Object: Key
            str,
            # Object: body. It's a list since we store the labelled
            # passages as JSONL.
            list[LabelledPassage],
        ],
    ],
) -> list[
    tuple[
        # Object: Key
        str,
        list[
            tuple[
                # Text block (aka span) ID
                str,
                VespaConcept,
            ]
        ],
    ]
]:
    """Convert labelled passages to document concepts for Vespa indexing."""
    return [
        (
            s3_path,
            convert_labelled_passages_to_concepts(labelled_passages=labelled_passages),
        )
        for s3_path, labelled_passages in document_labelled_passages
    ]


def s3_paths_or_s3_prefixes(
    classifier_specs: Optional[list[ClassifierSpec]],
    document_ids: Optional[list[str]],
    config: Config,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """
    Return the paths or prefix for the documents and classifiers.

    - s3_prefix: The S3 prefix (directory) to yield objects from. Example:
      "s3://cpr-sandbox-data-pipeline-cache/labelled_passages"
    - s3_paths: A list of S3 object keys to yield objects from. Example:
      [
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json",
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.legislative.10695.6015.json",
      ]
    """
    logger = get_run_logger()

    match (classifier_specs, document_ids):
        case (None, None):
            # Run on all documents, regardless of classifier
            logger.info("run on all documents, regardless of classifier")
            s3_prefix = "s3://" + os.path.join(  # type: ignore
                config.cache_bucket,  # type: ignore
                config.document_source_prefix,
            )
            return None, [s3_prefix]

        case (list(), None):
            # Run on all documents, for the specified classifier
            logger.info("run on all documents, for the specified classifier")
            s3_prefixes = [
                "s3://"
                + os.path.join(  # type: ignore
                    config.cache_bucket,  # type: ignore
                    config.document_source_prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                )
                for classifier_spec in classifier_specs
            ]
            return None, s3_prefixes

        case (list(), list()):
            # Run on specified documents, for the specified classifier
            logger.info("run on specified documents, for the specified classifier")
            document_paths = [
                "s3://"
                + os.path.join(  # type: ignore
                    config.cache_bucket,  # type: ignore
                    config.document_source_prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                    f"{doc_id}.json",
                )
                for classifier_spec in classifier_specs
                for doc_id in document_ids
            ]
            return document_paths, None

        case (None, list()):
            raise ValueError(
                "if document IDs are specified, a classifier "
                "specifcation must also be specified, since they're "
                "namespaced by classifiers (e.g. "
                "`s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/"
                "v4/CCLW.legislative.10695.6015.json`)"
            )


def s3_obj_generator(
    s3_prefixes: Optional[list[str]],
    s3_paths: Optional[list[str]],
) -> Generator[
    tuple[
        # Object: key
        str,
        # Object: body. It's a list since we store the labelled
        # passages as JSONL.
        list[dict],
    ],
    None,
    None,
]:
    """
    Return a generator that returns S3 objects for each path or prefix.

    These will be for each output from the inference stage, of
    labelled passages.
    """
    logger = get_run_logger()

    match (s3_prefixes, s3_paths):
        case (list(), list()):
            raise ValueError(
                "Either s3_prefixes or s3_paths must be provided, not both."
            )
        case (list(), None):
            logger.info("S3 object generator: prefixes")
            return s3_obj_generator_from_s3_prefixes(s3_prefixes=s3_prefixes)
        case (None, list()):
            logger.info("S3 object generator: paths")
            return s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
        case (None, None):
            raise ValueError("Either s3_prefix or s3_paths must be provided.")
        case (_, _):
            raise ValueError(
                f"Unexpected types: `s3_prefixes={type(s3_prefixes)}`, "
                f"`s3_paths={type(s3_paths)}`"
            )


@flow
async def index_by_s3(
    aws_env: AwsEnv,
    vespa_search_adapter: Optional[VespaSearchAdapter] = None,
    s3_prefixes: Optional[list[str]] = None,
    s3_paths: Optional[list[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    as_subflow=True,
) -> None:
    """
    Asynchronously index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    indexes them in a Vespa instance. The name of each file in the specified S3 path is
    expected to represent the document's import ID.

    When `s3_prefix` is provided, the function will index all files within that S3
    prefix (directory). When `s3_paths` is provided, the function will index only the
    files specified in the list of S3 object keys. If both are provided `s3_paths` will
    be used.

    Assumptions:
    - The S3 file names represent document import IDs.

    params:
    - s3_prefix: The S3 prefix (directory) to yield objects from.
        E.g. "s3://bucket/prefix/"
    - s3_paths: A list of S3 object keys to yield objects from.
        E.g. {"s3://bucket/prefix/file1.json", "s3://bucket/prefix/file2.json"}
    - vespa_search_adapter: An instance of VespaSearchAdapter.
        E.g. VespaSearchAdapter(
            instance_url="https://vespa-instance-url.com",
            cert_directory="certs/"
        )
    """
    logger = get_run_logger()

    logger.info("getting S3 object generator")
    s3_objects = s3_obj_generator(s3_prefixes, s3_paths)

    logger.info("getting S3 labelled passages generator")
    document_labelled_passages = labelled_passages_generator(generator_func=s3_objects)

    logger.info("converting labelled passages to Vespa concepts")

    document_labelled_passages_batches = iterate_batch(
        document_labelled_passages, batch_size=batch_size
    )

    for (
        document_labelled_passages_batch_num,
        document_labelled_passages_batch,
    ) in enumerate(document_labelled_passages_batches, start=1):
        logger.info(
            f"processing batch document labelled passages #{document_labelled_passages_batch_num}"
        )

        document_concepts = await convert_labelled_passages_to_document_concepts(
            document_labelled_passages_batch
        )

        # It's possible that if there were too many concepts, we need to split it,
        # and thus we may end up outside of the "original" batch size.
        document_concepts_maybe_split = split_large_concepts_updates(
            document_concepts, MAX_CONCEPTS_IN_PARTIAL_UPDATE
        )

        indexing_tasks = [
            run_partial_updates_of_concepts_for_document_passages_as(
                s3_key,
                document_concepts,
                as_subflow,
                aws_env=aws_env,
            )
            for s3_key, document_concepts in document_concepts_maybe_split
        ]

        logger.info(
            f"gathering indexing tasks for batch {document_labelled_passages_batch_num}"
        )
        results = await asyncio.gather(*indexing_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Get the s3_key for the failed task
                s3_key = document_concepts[i][0]

                logger.error(
                    f"failed to process document for S3 key `{s3_key}`: {str(result)}",
                )


def split_large_concepts_updates(
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            VespaConcept,
        ]
    ],
    max_concepts_in_partial_update: int,
) -> list[
    tuple[
        # Text block (aka span) ID
        str,
        VespaConcept,
    ]
]:
    """
    Split up a list of concepts into multiple lists of concepts.

    This is done to ensure that they fit into the max parameter size
    for Prefect flow runs.
    """

    def split(current, new):
        obj_key, concepts = new  # unpack

        if len(concepts) > max_concepts_in_partial_update:
            concepts_batches = iterate_batch(
                concepts, batch_size=max_concepts_in_partial_update
            )
            concepts_split = [
                (obj_key, concepts)
                for concepts in concepts_batches  # repack
            ]

            return current + concepts_split

        return current + [new]

    return reduce(split, document_concepts, [])


def run_partial_updates_of_concepts_for_document_passages_as(
    s3_key: str,
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            VespaConcept,
        ]
    ],
    as_subflow: bool,
    aws_env: AwsEnv,
) -> Awaitable:
    """Run partial updates for document passages, either as a subflow or directly."""
    document_import_id = Path(s3_key).stem

    if as_subflow:
        flow_name = function_to_flow_name(
            run_partial_updates_of_concepts_for_document_passages
        )
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)

        return run_deployment(
            name=f"{flow_name}/{deployment_name}",
            parameters={
                "document_import_id": document_import_id,
                "document_concepts": dump_document_concepts(document_concepts),
            },
            timeout=1200,
            as_subflow=True,
        )
    else:
        return run_partial_updates_of_concepts_for_document_passages(  # pyright: ignore[reportCallIssue]
            document_import_id=document_import_id,
            document_concepts=document_concepts,  # pyright: ignore[reportArgumentType]
        )


def dump_document_concepts(
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            VespaConcept,
        ]
    ],
):
    """Dump document concepts for serialisation."""
    return [
        (text_block_id, concept.model_dump_json())
        for text_block_id, concept in document_concepts
    ]


def load_document_concepts(
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            # JSON string representation of VespaConcept
            str,
        ]
    ],
) -> list[tuple[str, VespaConcept]]:
    """Load document concepts from serialised JSON back into VespaConcept objects."""
    return [
        (text_block_id, VespaConcept.model_validate_json(concept_json))
        for text_block_id, concept_json in document_concepts
    ]


def maybe_load_document_concepts(
    document_concepts: list[
        tuple[
            # Text block (aka span) ID
            str,
            Union[
                VespaConcept,
                # Serialised JSON of object
                str,
            ],
        ]
    ],
) -> list[
    tuple[
        # Text block (aka span) ID
        str,
        VespaConcept,
    ]
]:
    """Maybe load document concepts from serialised JSON back into VespaConcept objects."""
    # Nothing to do if there's none
    if len(document_concepts) == 0:
        return document_concepts  # pyright: ignore[reportReturnType]

    # Based on the first document concept, if it's a string, then
    # deserialise all of them
    (text_block_id, vespa_concept) = document_concepts[0]
    if isinstance(vespa_concept, str):
        return load_document_concepts(document_concepts)  # pyright: ignore[reportArgumentType]

    return document_concepts  # pyright: ignore[reportReturnType]


def iterate_batch(
    data: Union[list[Any], Generator[Any, None, None]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Generator[
    list[Any],
    None,
    None,
]:
    """Generate batches from a list or generator with a specified size."""
    if isinstance(data, list):
        # For lists, we can use list slicing
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]
    else:
        # For generators, accumulate items until we reach batch size
        batch = []
        for item in data:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget to yield the last partial batch
            yield batch


@flow
async def index_labelled_passages_from_s3_to_vespa(
    classifier_specs: Optional[list[ClassifierSpec]] = None,
    document_ids: Optional[list[str]] = None,
    config: Optional[Config] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Asynchronously index concepts from S3 into Vespa.

    This function retrieves concept documents from files stored in an
    S3 path and indexes them in a Vespa instance. The name of each
    file in the specified S3 path is expected to represent the
    document's import ID.
    """
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")

    logger.info(f"running with config: {config}")

    if classifier_specs is None:
        logger.info("no classifier specs. passed in, loading from file")
        classifier_specs = parse_spec_file(config.aws_env)

    logger.info(f"running with classifier specs.: {classifier_specs}")
    s3_paths, s3_prefixes = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config,
    )

    logger.info(f"s3_prefix: {s3_prefixes}, s3_paths: {s3_paths}")

    await index_by_s3(
        aws_env=config.aws_env,
        vespa_search_adapter=config.vespa_search_adapter,  # type: ignore
        s3_prefixes=s3_prefixes,
        s3_paths=s3_paths,
        batch_size=batch_size,
        as_subflow=config.as_subflow,
    )
