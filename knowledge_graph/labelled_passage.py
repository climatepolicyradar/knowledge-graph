import html
import logging
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from knowledge_graph.identifiers import Identifier, WikibaseID
from knowledge_graph.span import Span, merge_overlapping_spans

logger = logging.getLogger(__name__)


class LabelledPassage(BaseModel):
    """Represents a passage of text which has been labelled by an annotator"""

    id: str = Field(..., title="ID", description="The unique identifier of the passage")
    text: str = Field(..., title="Text", description="The text of the passage")
    spans: list[Span] = Field(
        default_factory=list,
        title="Spans",
        description="The spans in the passage which have been labelled by the annotator",
        repr=False,
    )
    metadata: dict = Field(
        default_factory=dict,
        title="Metadata",
        description="Additional data, eg translation status or dataset",
        repr=False,
    )

    def __init__(self, text: str, spans: list[Span], **kwargs):
        id = kwargs.pop("id", Identifier.generate(text))
        super().__init__(text=text, spans=spans, id=id, **kwargs)

    @model_validator(mode="after")
    def check_whether_spans_are_within_text(self):
        """Check whether the spans are within the text"""
        for span in self.spans:
            if span.end_index > len(self.text):
                raise ValueError(
                    "end_index must be less than or equal to the length of the text"
                )
        return self

    def get_highlighted_text(
        self, start_pattern: str = "[cyan]", end_pattern: str = "[/cyan]"
    ) -> str:
        """
        Returns the text with highlighted spans, usable in a rich console output

        :param str start_pattern: The pattern to add to the text at the start of a span
        :param str end_pattern: The pattern to add to the text at the end of a span
        :return str: The text with highlighted spans
        """
        # Decode HTML entities
        decoded_text = html.unescape(self.text)
        # remove all html tags
        decoded_text = re.sub(r"<[^>]*>", "", decoded_text)

        # merge any overlapping spans
        merged_spans = merge_overlapping_spans(self.spans)

        # create the output text
        output = ""
        last_end = 0
        sorted_spans = sorted(merged_spans, key=lambda span: span.start_index)
        for span in sorted_spans:
            # Add text before the span
            output += decoded_text[last_end : span.start_index]
            # Add highlighted span
            output += f"{start_pattern}{decoded_text[span.start_index : span.end_index]}{end_pattern}"
            last_end = span.end_index

        # Add any remaining text after the last span
        output += decoded_text[last_end:]

        return output

    @property
    def labellers(self) -> list[str]:
        """
        Returns the set of labellers who labelled the passage

        :return list[str]: The set of labellers who labelled the passage
        """
        return list(
            set([labeller for span in self.spans for labeller in span.labellers])
        )

    def __eq__(self, other: object) -> bool:
        """Check whether two labelled passages are equal"""
        if not isinstance(other, LabelledPassage):
            return False
        if self.text != other.text:
            return False
        if set(self.spans) != set(other.spans):
            return False
        return True

    @property
    def sanitised_text(self) -> str:
        """A normalised version of the text which can be used for comparison"""
        return self.sanitise(self.text)

    @staticmethod
    def sanitise(text: str) -> str:
        """Sanitise text by replacing bad XML characters and normalizing text."""
        # First handle XML special characters
        bad_xml_strings = ["&", "<", ">", '"', "'"]
        xml_translation = str.maketrans(
            {string: "_" * len(string) for string in bad_xml_strings}
        )
        text = text.translate(xml_translation)

        # Then normalize common Unicode discrepancies and whitespace variations
        normalize_translation = str.maketrans(
            {
                " ": " ",
                "\n": " ",
                "\t": " ",
                "…": "...",
                "'": "'",
                "—": "-",
                "’": "'",
                "“": '"',
                "”": '"',
            }
        )
        return text.translate(normalize_translation)

    def __hash__(self) -> int:
        """Hash based on id for use in sets/dicts"""
        return hash(self.id)


def consolidate_passages_by_text(
    labelled_passages: list[LabelledPassage],
) -> list[LabelledPassage]:
    """
    Merge multiple LabelledPassages for the same text into single LabelledPassages.

    This function combines multiple passages with the same text into single
    passages, merging spans from all labellers. This is useful when pulling
    passages from Argilla where multiple labellers may have annotated the same
    text, resulting in separate passages per response.

    Metadata from the first passage in each group is preserved (since passages with
    identical text should have the same metadata from the same source document).

    Args:
        labelled_passages: List of passages, potentially containing duplicates
            with the same text but different spans from different labellers.

    Returns:
        List of merged_passages passages with unique texts and merged spans.

    Example:
        >>> # Three passages: same text, different labellers
        >>> passage1 = LabelledPassage(text="Hello world", spans=[span_a])
        >>> passage2 = LabelledPassage(text="Hello world", spans=[span_b])
        >>> passage3 = LabelledPassage(text="Different", spans=[span_c])
        >>> merged = consolidate_passages_by_text([passage1, passage2, passage3])
        >>> len(merged)  # Two unique texts
        2
        >>> len(merged[0].spans)  # Spans from both labellers combined
        2
    """
    passage_groups: dict[str, list[LabelledPassage]] = defaultdict(list)
    for passage in labelled_passages:
        passage_groups[passage.id].append(passage)

    merged_passages: list[LabelledPassage] = []
    for group in passage_groups.values():
        merged_passages.append(
            LabelledPassage(
                text=group[0].text,
                spans=[span for passage in group for span in passage.spans],
                metadata=group[0].metadata,
            )
        )
    return merged_passages


def labelled_passages_to_dataframe(
    labelled_passages: list[LabelledPassage],
) -> pd.DataFrame:
    """
    Convert a list of labelled passages to a dataframe, with the following characteristics:

    - spans are not exploded, i.e. they are left as JSON
    - metadata fields are given one column each
    - new columns `prediction` and `prediction_probability` are added
    """

    labelled_passages_as_dicts = [lp.model_dump() for lp in labelled_passages]

    boolean_predictions = [bool(lp.spans) for lp in labelled_passages]

    if all(
        [
            span.prediction_probability is None
            for lp in labelled_passages
            for span in lp.spans
        ]
    ):
        prediction_probabilities = [None] * len(labelled_passages)
    else:
        prediction_probabilities = [
            max([span.prediction_probability or 0 for span in lp.spans])
            if lp.spans
            else 0
            for lp in labelled_passages
        ]

    df = pd.json_normalize(labelled_passages_as_dicts)
    df["prediction"] = boolean_predictions
    df["prediction_probability"] = prediction_probabilities

    return df


def _detect_human_label_format(value) -> bool | None:
    """
    Detect whether a human label value indicates a positive or negative label.

    Supports multiple formats:
    - Boolean: True (positive) / False (negative)
    - String: "yes"/"y"/"true"/"1" (positive), "no"/"n"/"false"/"0" (negative)
    - Numeric: non-zero (positive) / 0 (negative)
    - Empty/NaN: None (no label - absence of annotation)

    :param value: The value from the human label column
    :return bool | None: True for positive label, False for negative label, None for no label
    """
    # Empty values - return None (i.e. no label/absence of annotation)
    if pd.isna(value) or value == "" or value is None:
        return None

    if isinstance(value, bool):
        return value

    # String values
    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in ["yes", "y", "true", "1"]:
            return True
        elif value_lower in ["no", "n", "false", "0"]:
            return False

        # Any other non-empty string is treated as True
        return True

    # Numeric values: 0 is False, non-zero is True
    try:
        numeric_value = float(value)
        return numeric_value != 0
    except (ValueError, TypeError):
        return None


def _reconstruct_metadata(df_row: pd.Series) -> dict:
    """
    Reconstruct nested metadata dictionary from flattened DataFrame columns.

    Converts columns like 'metadata.source' and 'metadata.dataset' back to
    {'source': ..., 'dataset': ...}

    :param pd.Series df_row: A row from the DataFrame
    :return dict: The reconstructed metadata dictionary
    """
    metadata = {}
    for col_name in df_row.index:
        col_str = str(col_name)
        if col_str.startswith("metadata."):
            key = col_str[len("metadata.") :]
            value = df_row[col_name]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                metadata[key] = value
    return metadata


def dataframe_to_labelled_passages(
    df: pd.DataFrame,
    concept_id: WikibaseID,
    labeller_names: list[str] = [],
    human_label_column: str | None = None,
    skip_unlabelled: bool = True,
) -> list[LabelledPassage]:
    """
    Convert a DataFrame to a list of LabelledPassages with human-labelled spans.

    Only spans from human labels are created. Any spans that were in the 'spans' column
    of the dataframe are discarded.

    :param pd.DataFrame df: Input DataFrame with columns:
        - 'text': passage text (required)
        - 'id': passage identifier (required)
        - 'metadata.*': flattened metadata fields (optional, e.g., 'metadata.source')
        - human label column specified by `human_label_column` parameter
    :param concept_id: Concept ID to use for human-labelled spans
    :param labeller_names: List of identifiers for the human labellers. All names will be
        added to each span's labellers field. If not provided, spans will have empty
        labellers lists.
    :param human_label_column: Name of column containing human labels. If not specified,
        passages will be created with no spans.
    :param skip_unlabelled: If True, skip passages where the human label is None
        (empty/unlabeled). Only passages with explicit labels (True or False) will be
        returned. Defaults to True.
    :return list[LabelledPassage]: List of LabelledPassage objects with human-labelled spans

    :Example:

        >>> df = pd.DataFrame({
        ...     'text': ['Climate change is severe', 'Another passage'],
        ...     'human_label': [True, False],
        ...     'metadata.source': ['doc1', 'doc1']
        ... })
        >>> passages = dataframe_to_labelled_passages(
        ...     df,
        ...     labeller_names=['alice', 'bob'],
        ...     human_label_column='human_label',
        ...     default_concept_id='Q42'
        ... )
        >>> assert len(passages[0].spans) == 1  # Positive label creates span
        >>> assert len(passages[1].spans) == 0  # Negative label, no span
        >>> assert passages[0].spans[0].labellers == ['alice', 'bob']
    """

    if bool(df["id"].isnull().any()):
        len_before = len(df)
        df = df.dropna(subset="id")
        n_rows_dropped = len_before - len(df)

        logger.warning(
            f"Input dataframe does not have values for all rows in the 'id' column.\nDropped {n_rows_dropped} with no ID value."
        )

    passages = []

    for _, row in df.iterrows():
        passage_id = row["id"]
        text = str(row["text"])
        metadata = _reconstruct_metadata(row)

        spans = []

        if human_label_column and human_label_column in row.index:
            label = _detect_human_label_format(row[human_label_column])

            if skip_unlabelled and label is None:
                continue

            # Only create a span for explicit positive labels (True)
            # False = explicit negative label (no span created)
            # None = no label/unlabeled (no span created)
            if label is True:
                timestamp = datetime.now()
                timestamps = [timestamp] * len(labeller_names)

                spans.append(
                    Span(
                        text=text,
                        start_index=0,
                        end_index=len(text),
                        concept_id=WikibaseID(concept_id),
                        labellers=labeller_names,
                        timestamps=timestamps,
                    )
                )

        passage = LabelledPassage(
            text=text, spans=spans, metadata=metadata, id=passage_id
        )

        passages.append(passage)

    return passages
