import json
from datetime import date
from pathlib import Path

import boto3
from datasets import load_dataset

from flows.vibe_check import LabelledPassageWithMarkup, _get_bucket_name_from_ssm
from knowledge_graph.classifier import Classifier
from knowledge_graph.concept import Concept
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.utils import serialise_pydantic_list_as_jsonl

# This example pulls 100 passages from the HuggingFace dataset but you could use any
# set of passages you like, as long as they come with the appropriate text and
# metadata fields.
sample = load_dataset(
    "ClimatePolicyRadar/all-document-text-data", split="train[:100]"
).to_pandas()  # type: ignore
texts = sample["text_block.text"].tolist()  # type: ignore

your_custom_classifier = Classifier.load("path/to/your/classifier.pkl")
classifier_id = your_custom_classifier.id
classifier_name = your_custom_classifier.name
concept: Concept = your_custom_classifier.concept

# Run inference
predicted_spans_list = your_custom_classifier.predict(texts)
labelled_passages = [
    LabelledPassage(
        text=text,
        spans=spans,
        metadata={str(k): str(v) for k, v in row.to_dict().items()},
    )
    for text, spans, (_, row) in zip(texts, predicted_spans_list, sample.iterrows())
]

# Add HTML markup to the labelled passages and serialize to JSONL
passages_with_markup = [
    LabelledPassageWithMarkup.from_labelled_passage(p) for p in labelled_passages
]
jsonl_string = serialise_pydantic_list_as_jsonl(passages_with_markup)

# Upload to S3
assert concept.wikibase_id is not None
prefix = Path(concept.wikibase_id) / classifier_id
bucket_name = _get_bucket_name_from_ssm()
s3 = boto3.Session(region_name="eu-west-1", profile_name="labs").client("s3")

# We need to upload three things to s3:
# Metadata about the concept
s3.put_object(
    Bucket=bucket_name,
    Key=str(prefix / "concept.json"),
    Body=concept.model_dump_json().encode(),
)
# Metadata about the classifier
s3.put_object(
    Bucket=bucket_name,
    Key=str(prefix / "classifier.json"),
    Body=json.dumps(
        {
            "id": classifier_id,
            "name": classifier_name,
            "date": date.today().isoformat(),
        }
    ).encode(),
)
# The predictions
s3.put_object(
    Bucket=bucket_name,
    Key=str(prefix / "predictions.jsonl"),
    Body=jsonl_string.encode(),
)
