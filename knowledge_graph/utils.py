"""Utility functions for the knowledge_graph package."""

import json
import logging
from collections.abc import Generator, Sequence
from typing import Any, TypeVar

import prefect
import prefect.exceptions
import prefect.logging
import typer
from pydantic import BaseModel, ValidationError
from rich.logging import RichHandler

LoggingAdapter = logging.LoggerAdapter[logging.Logger]


def get_logger(name: str = "knowledge_graph") -> logging.Logger | LoggingAdapter:
    """
    Get a logger via Prefect.

    You can overwrite the logging level[2]. If not running in a flow
    or task run context, a logger that doesn't send to the Prefect API
    is returned.

    > `get_run_logger()` can only be used in the context of a flow or task.
    > To use a normal Python logger anywhere with your same configuration, use `get_logger()` from `prefect.logging`.
    > The logger retrieved with `get_logger()` will not send log records to the Prefect API.

    [1]: https://docs.prefect.io/v3/how-to-guides/workflows/add-logging
    [2]: https://docs.prefect.io/v3/api-ref/settings-ref#logging-level
    """
    try:
        return prefect.logging.get_run_logger()
    except prefect.exceptions.MissingContextError:
        root = logging.getLogger("knowledge_graph")
        if not root.handlers:
            root.addHandler(RichHandler())
        if not root.level:
            root.setLevel(logging.INFO)
        return logging.getLogger(name)


T = TypeVar("T")


def parse_kwargs_from_strings(key_value_strings: list[str] | None) -> dict[str, Any]:
    """Parse key=value strings into dicts that can be used as kwargs."""
    if not key_value_strings:
        return {}

    kwargs = {}
    for kv in key_value_strings:
        if "=" not in kv:
            raise typer.BadParameter(
                f"Invalid format for kwarg: '{kv}'. Expected key=value format."
            )

        key, value = kv.split("=", 1)

        # Try to parse as int, then bool, then string
        try:
            kwargs[key] = int(value)
        except ValueError:
            if value.lower() in ("true", "false"):
                kwargs[key] = value.lower() == "true"
            else:
                kwargs[key] = value

    return kwargs


def iterate_batch(
    data: Sequence[T] | Generator[T, None, None],
    batch_size: int,
) -> Generator[Sequence[T], None, None]:
    """Generate batches from a list or generator with a specified size."""
    if isinstance(data, Sequence):
        # For lists, we can use list slicing
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]
    else:
        # For generators, accumulate items until we reach batch size
        batch: list[T] = []
        for item in data:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget to yield the last partial batch
            yield batch


def serialise_pydantic_list_as_jsonl[T: BaseModel](models: Sequence[T]) -> str:
    """
    Serialize a list of Pydantic models as JSONL (JSON Lines) format.

    Each model is serialized on a separate line using model_dump_json().
    """
    jsonl_content = "\n".join(model.model_dump_json() for model in models)

    return jsonl_content


def deserialise_pydantic_list_from_jsonl[T: BaseModel](
    jsonl_content: str, model_class: type[T]
) -> list[T]:
    """
    Deserialize JSONL (JSON Lines) format to a list of Pydantic models.

    Each line should contain a JSON object that can be parsed by the model_class.
    """
    models = []
    for line in jsonl_content.strip().split("\n"):
        if line.strip():  # Skip empty lines
            model = model_class.model_validate_json(line)
            models.append(model)
    return models


def deserialise_pydantic_list_with_fallback[T: BaseModel](
    content: str, model_class: type[T]
) -> list[T]:
    """
    Deserialize content to a list of Pydantic models with fallback support.

    First tries JSONL format, then falls back to original format (JSON array of JSON strings).
    """
    # Try JSONL format first
    try:
        return deserialise_pydantic_list_from_jsonl(content, model_class)
    except ValidationError:
        # Fall back to original format (array of JSON strings)
        data = json.loads(content)
        return [model_class.model_validate_json(passage) for passage in data]
