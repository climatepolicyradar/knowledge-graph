"""
Add jurisdiction information to the litigation dataset.

Usage:
poetry run python scripts/sampling_for_sectors_classifier/add_geography_to_litigation.py
"""

import json

import pandas as pd
from cpr_sdk.parser_models import BaseParserOutput
from rich.console import Console
from rich.progress import track

from scripts.config import interim_data_dir, processed_data_dir, raw_data_dir
from src.identifiers import generate_identifier

console = Console()


litigation_us_csv_path = raw_data_dir / "litigation-us.csv"
litigation_non_us_csv_path = raw_data_dir / "litigation-non-us.csv"

litigation_us_df = pd.read_csv(litigation_us_csv_path)
litigation_non_us_df = pd.read_csv(litigation_non_us_csv_path)
litigation_df = pd.concat([litigation_us_df, litigation_non_us_df]).dropna()

litigation_df["Jurisdictions"] = litigation_df["Jurisdictions"].str.split(">").str[0]

# Add an identifier to each document
litigation_df["id"] = litigation_df.apply(
    lambda x: generate_identifier(input_string=x["Title"] + x["Document file"]),
    axis=1,
)
litigation_df.set_index("id", inplace=True)


translated_litigation_documents_path = interim_data_dir / "translated" / "litigation"

output_dir = processed_data_dir / "documents" / "litigation"
output_dir.mkdir(parents=True, exist_ok=True)
for document_path in track(
    list(translated_litigation_documents_path.glob("*.json")),
    description="Adding jurisdiction information",
):
    litigation_document = json.loads(document_path.read_text(encoding="utf-8"))
    parser_output = BaseParserOutput(**litigation_document)

    document_id = document_path.stem
    parser_output.document_metadata["geography"] = litigation_df.loc[
        document_id, "Jurisdictions"
    ]

    output_path = output_dir / f"{document_id}.json"
    output_path.write_text(parser_output.model_dump_json())
