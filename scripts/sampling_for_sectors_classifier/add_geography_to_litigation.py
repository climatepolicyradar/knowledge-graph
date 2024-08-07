"""
Add jurisdiction information to the litigation dataset.

Usage:
poetry run python scripts/sampling_for_sectors_classifier/add_geography_to_litigation.py
"""

import json
from pathlib import Path

import pandas as pd
from cpr_data_access.parser_models import BaseParserOutput
from rich.console import Console
from rich.progress import track

from src.identifiers import generate_identifier

console = Console()

data_dir = Path("data")
litigation_us_csv_path = data_dir / "raw" / "litigation-us.csv"
litigation_non_us_csv_path = data_dir / "raw" / "litigation-non-us.csv"

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


translated_litigation_documents_path = (
    data_dir / "interim" / "translated" / "litigation"
)

output_dir = data_dir / "processed" / "documents" / "litigation"
output_dir.mkdir(parents=True, exist_ok=True)
for document_path in track(
    list(translated_litigation_documents_path.glob("*.json")),
    description="Adding jurisdiction information",
):
    with open(document_path, "r") as f:
        litigation_document = json.load(f)
    parser_output = BaseParserOutput(**litigation_document)

    document_id = document_path.stem
    parser_output.document_metadata["geography"] = litigation_df.loc[
        document_id, "Jurisdictions"
    ]

    output_path = output_dir / f"{document_id}.json"
    output_path.write_text(parser_output.model_dump_json())
