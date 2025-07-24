import json
import os
from pathlib import Path

import boto3
import pandas as pd
from cpr_sdk.models import BaseDocument
from cpr_sdk.parser_models import BaseParserOutput
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.classifier.large_language_model import DEFAULT_SYSTEM_PROMPT, LLMClassifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage

console = Console(highlight=False)


session = boto3.Session(profile_name="labs", region_name="eu-west-1")
ssm = session.client("ssm")
api_key = ssm.get_parameter(Name="OPENAI_API_KEY", WithDecryption=True)["Parameter"][
    "Value"
]
os.environ["OPENAI_API_KEY"] = api_key
console.print("🔑 Loaded OpenAI API key from SSM")

concept = Concept(
    wikibase_id="Q99999999",
    preferred_label="document mention",
    alternative_labels=[
        "citation",
        "referencereference to a document",
        "document reference",
        "something that looks like a reference to a document",
    ],
    description="The title of a document",
    definition="""
        A reference to a document which might appear somewhere in the climate policy radar database.
        That could be a formal citation or a loose, off-hand reference.
        These should NOT be the names of governmental bodies or organisations, acronyms, the names of sub-articles, frameworks, or conventions, or other names which are not the title of a document.
        Correct examples: 'The Paris Agreement', 'the 2015 Paris Agreement', 'Kenya's NDC', 'Singapore's First BTR'
        Incorrect examples: 'Article 10', 'UNFCCC', 'Paris', 'COP-28', 'Chapter 3'
    """,
)
console.print(f"🤓 Created a concept: {concept}")

# you can override the LLM's system prompt here, if you want to augment the instructions
custom_system_prompt_template = (
    DEFAULT_SYSTEM_PROMPT
    + "You should be very conservative in your judgements, only labelling stuff which you're sure is a reference to a document."
)

classifier = LLMClassifier(
    concept=concept,
    model_name="gpt-4.1-mini",
    system_prompt_template=custom_system_prompt_template,
)
console.print(f"🤖 Created a {classifier}")

parser_output_dir = Path("data/processed/documents/translated_parser_outputs")
parser_output_paths = list(parser_output_dir.rglob("*.json"))

documents: list[BaseDocument] = []
for parser_output_path in parser_output_paths:
    with open(parser_output_path, encoding="utf-8") as f:
        parser_output_data = json.load(f)
        parser_output = BaseParserOutput(**parser_output_data)
        document = BaseDocument.from_parser_output(parser_output)
        documents.append(document)
console.print(f"📚 Loaded {len(documents)} documents from {parser_output_dir}")

document_passage_dict = {
    document.document_id: {
        "document_name": document.document_name,
        "document_source_url": document.document_source_url,
        "passages": [
            text_block.to_string()
            for text_block in document.text_blocks
            if len(text_block.to_string()) > 20
        ],
    }
    for document in documents
}
n_passages = sum(
    [len(document["passages"]) for document in document_passage_dict.values()]
)
console.print(f"📝 Gathered {n_passages} passages from {len(documents)} documents")

labelled_passages: list[LabelledPassage] = []
batch_size = 50
with Progress(
    TextColumn("[progress.description]{task.description}"),
    MofNCompleteColumn(),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console,
    transient=True,
) as progress_bar:
    task = progress_bar.add_task(
        f"Labeling passages with {classifier.name}", total=n_passages
    )
    for document_id, document_data in document_passage_dict.items():
        for i in range(0, len(document_data["passages"]), batch_size):
            batch_passages = document_data["passages"][i : i + batch_size]
            batch_spans = classifier.predict_batch(batch_passages)

            for passage_text, predicted_spans in zip(batch_passages, batch_spans):
                try:
                    labelled_passage = LabelledPassage(
                        text=passage_text,
                        spans=predicted_spans,
                        concept_id=concept.wikibase_id,
                        metadata={
                            "document_id": document_id,
                            "document_name": document_data["document_name"],
                            "document_source_url": document_data["document_source_url"],
                        },
                    )
                except Exception as e:
                    console.print(f"[red]Error labelling passage[/red]: {e}")
                    labelled_passage = LabelledPassage(
                        text=passage_text,
                        spans=[],
                        concept_id=concept.wikibase_id,
                    )
                labelled_passages.append(labelled_passage)

            n_positive_passages = sum(
                # count the number of labelled passages which contain spans
                [bool(passage.spans) for passage in labelled_passages]
            )

            # print the labelled passages which contain spans
            for labelled_passage in labelled_passages:
                if labelled_passage.spans:
                    console.print(
                        labelled_passage.get_highlighted_text(), end="\n---\n"
                    )

            progress_bar.update(
                task,
                advance=len(batch_passages),
                description=f"found {n_positive_passages} passages with document mentions",
            )

n_positive_passages = sum([bool(passage.spans) for passage in labelled_passages])
n_spans = sum([len(passage.spans) for passage in labelled_passages])
console.print(
    f"🤓 Found {n_spans} spans in {n_positive_passages}/{n_passages} passages"
)


rows = []
for labelled_passage in labelled_passages:
    for span in labelled_passage.spans:
        rows.append(
            {
                "text": span.text,
                "labelled_text": span.labelled_text,
                **labelled_passage.metadata,
            }
        )
df = pd.DataFrame(rows)
df.to_csv("document_mentions.csv", index=False)
console.print("💾 Saved labelled spans to data/labelled_spans.csv")
