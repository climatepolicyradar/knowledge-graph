import json
import os
from datetime import datetime

# import argilla.v1 as rg_v1
import huggingface_hub as hf
import pandas as pd
import typer
from datasets import DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from rich.console import Console

# from src.argilla_legacy import concept_to_dataset_name
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

load_dotenv(find_dotenv())


app = typer.Typer()
console = Console()


def relevant_spans(hf_spans: list[list], concept_id: WikibaseID | None) -> list[list]:
    """Filters the spans that are irrelevent for the given concept"""
    concept_to_label_map = {
        "Q1651": {"NZT", "Reduction", "Other", "Conditional"},
        "Q1652": {"NZT", "Reduction"},
        "Q1653": {"NZT"},
    }

    relevant_labels = concept_to_label_map[str(concept_id)]
    return [span for span in hf_spans if span[0] in relevant_labels]


def span_from_hf_span(
    hf_span: list,
    text: str,
    concept_id: WikibaseID | None,
    labellers: list[str],
    timestamps: list[datetime],
) -> Span:
    """
    Creates a Span object from a HF span

    HF spans are of the form [label, start, end].
    """
    _, start, end = hf_span

    return Span(
        text=text,
        start_index=start,
        end_index=end,
        concept_id=concept_id,
        labellers=labellers,
        timestamps=timestamps,
    )


def labelled_passage_from_row(
    row: pd.Series, concept_id: WikibaseID | None
) -> LabelledPassage:
    """Turns the row from the HF targets dataset into LabelledPassage objects"""
    hf_spans = relevant_spans(json.loads(row["annotation"]), concept_id)
    text = row["text"]

    spans = [
        span_from_hf_span(
            hf_span=hf_span,
            text=text,
            concept_id=concept_id,
            labellers=[row["annotation_agent"]],
            timestamps=[row["event_timestamp"]],
        )
        for hf_span in hf_spans
    ]

    return LabelledPassage(text=text, spans=spans, metadata=json.loads(row["metadata"]))


# NOTE: this script was run with legacy argilla v1.
# Since the legacy handling has been removed, this script will not work as is, but is still retained
# for illustration purposes.


@app.command()
def main():
    # getting the target concepts
    # session = WikibaseSession(
    #     os.getenv("WIKIBASE_USERNAME"),
    #     os.getenv("WIKIBASE_PASSWORD"),
    #     os.getenv("WIKIBASE_URL"),
    # )

    # concepts = session.get_concepts(wikibase_ids=["Q1651", "Q1652", "Q1653"])  # type: ignore

    # getting the annotations from HF
    hf.login(token=os.getenv("HF_TOKEN"))
    ds = load_dataset("ClimatePolicyRadar/targets")
    assert isinstance(ds, DatasetDict)
    df = ds["train"].to_pandas()
    assert isinstance(df, pd.DataFrame)
    console.log(f"✅ Loaded {len(df)} labelled passages from HF")

    # wrangling and uploading to Argilla
    # rg_v1.init(
    #     api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL")
    # )

    # workspace_name = "knowledge-graph"  # after discussing with Harrison, this is where everything should go

    # try:
    #     workspace = rg_v1.Workspace.create(name=workspace_name)
    #     console.log(f'✅ Created workspace "{workspace.name}"')
    # except ValueError:
    #     workspace = rg_v1.Workspace.from_name(name=workspace_name)
    #     console.log(f'✅ Loaded workspace "{workspace.name}"')

    # usernames = set(df["annotation_agent"].tolist())

    # for username in usernames:
    #     try:
    #         password = generate_identifier(username)
    #         _ = rg_v1.User.create(
    #             username=username,
    #             password=password,
    #             role="annotator",  # type: ignore
    #         )
    #         console.log(f'✅ Created user "{username}" with password "{password}"')
    #     except KeyError:
    #         console.log(f'✅ User "{username}" already exists')
    #         _ = rg_v1.User.from_name(username)

    # for concept in concepts:
    #     target_labelled_passages = [
    #         labelled_passage_from_row(row, concept.wikibase_id)
    #         for _, row in df.iterrows()
    #     ]
    #     # copied this over from scripts/push_to_argilla.py
    #     dataset = FeedbackDataset(
    #         guidelines="Highlight the entity if it is present in the text",
    #         fields=[
    #             TextField(name="text", title="Text", use_markdown=True),  # type: ignore
    #         ],
    #         questions=[
    #             SpanQuestion(  # type: ignore
    #                 name="entities",
    #                 labels={concept.wikibase_id: concept.preferred_label},
    #                 field="text",
    #                 required=True,
    #                 allow_overlapping=False,
    #             )
    #         ],
    #     )

    #     records = []

    #     for passage in target_labelled_passages:
    #         record = FeedbackRecord(
    #             fields={"text": passage.text},  # type: ignore
    #             metadata=passage.metadata,
    #             responses=[
    #                 ResponseSchema(
    #                     values=[
    #                         {
    #                             "entities": ValueSchema(
    #                                 value=[
    #                                     SpanValueSchema(
    #                                         label=concept.wikibase_id,
    #                                         start=span.start_index,
    #                                         end=span.end_index,
    #                                     )
    #                                 ]
    #                             )
    #                         }
    #                     ]  # type: ignore
    #                 )
    #                 for span in passage.spans
    #             ],
    #         )
    #         records.append(record)

    #     dataset.add_records(records)  # type: ignore

    # dataset_name = concept_to_dataset_name(concept)

    # dataset_in_argilla = dataset.push_to_argilla(
    #     name=dataset_name, workspace=workspace_name, show_progress=False
    # )
    # console.log(f'✅ Created dataset for "{concept}" at {dataset_in_argilla.url}')


if __name__ == "__main__":
    app()
