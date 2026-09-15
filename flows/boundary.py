"""Boundary between Prefect and AWS."""

import gc
import json
import logging
from collections.abc import Sequence
from datetime import timedelta
from io import BytesIO
from typing import (
    Final,
    NewType,
    TypeVar,
)

from cpr_sdk.s3 import _s3_object_read_text
from pydantic import BaseModel, PositiveInt
from types_aiobotocore_s3.client import S3Client

from flows.utils import (
    DocumentObjectUri,
    S3Uri,
)
from knowledge_graph.labelled_passage import LabelledPassage

# Provide a generic type to use instead of `Any` for types hints
T = TypeVar("T")

CONCEPT_COUNT_SEPARATOR: Final[str] = ":"
DEFAULT_DOCUMENTS_BATCH_SIZE: Final[PositiveInt] = 50
DEFAULT_TEXT_BLOCKS_BATCH_SIZE: Final[PositiveInt] = 20
DEFAULT_UPDATES_TASK_BATCH_SIZE: Final[PositiveInt] = 5

# Get more logs
logging.basicConfig(level=logging.DEBUG)

# Set the garbage collection debugging flags. Debugging information will be written to sys.stderr. See below for a list of debugging flags which can be combined using bit operations to control debugging.
gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)

# The "parent" AKA the higher level flows that do multiple things.
PARENT_TIMEOUT_S: int = int(timedelta(hours=4).total_seconds())


class S3Accessor(BaseModel):
    """Representing S3 paths and prefixes for accessing documents."""

    paths: Sequence[str] | None = None
    prefixes: Sequence[str] | None = None

    def __str__(self) -> str:
        """String representation of the S3Accessor for logging"""
        prefix_count = len(self.prefixes) if self.prefixes else 0
        path_count = len(self.paths) if self.paths else 0
        return f"(prefixes={prefix_count}, paths={path_count})"

    def __repr__(self) -> str:
        """String representation of the S3Accessor for logging"""
        return self.__str__()


# AKA LabelledPassage
# Example: 18593
TextBlockId = NewType("TextBlockId", str)


async def s3_object_write_text_async(s3: S3Client, s3_uri: S3Uri, text: str) -> None:
    """Put an object in S3, async."""
    body = BytesIO(text.encode("utf-8"))
    await s3.put_object(
        Bucket=s3_uri.bucket,
        Key=s3_uri.key,
        Body=body,
        ContentType="application/json",
    )


async def s3_copy_file(s3: S3Client, source: S3Uri, target: S3Uri) -> None:
    """Copy a file from one S3 location to another."""
    await s3.copy_object(
        Bucket=source.bucket,
        CopySource=source.uri,
        Key=target.key,
    )


def load_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
) -> list[LabelledPassage]:
    """Load and transforms the S3 object's body into LabelledPassages objects."""
    object_json = json.loads(_s3_object_read_text(s3_path=document_object_uri))
    if len(object_json) == 0:
        return []

    # We had a window where we hadn't serialised the labelled
    # passages correctly, and needed this special handling.
    #
    # This has now been fixed[1], and in the near future this can be removed.
    #
    # [1] https://linear.app/climate-policy-radar/issue/PLA-505/labelled-passage-serialisation-varies-in-format-and-should-be-the-same
    if isinstance(object_json[0], str):
        object_json = [json.loads(labelled_passage) for labelled_passage in object_json]

    return [LabelledPassage(**labelled_passage) for labelled_passage in object_json]
