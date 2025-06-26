"""
Add jurisdiction information to the litigation dataset.

The script assumes that a set of translated document objects are stored in
`data/interim/translated`. The output will be saved in `data/processed/documents`.

The script uses the `data/raw/geography-iso-3166-backend.csv` file from the CPR backend
to map jurisdiction strings to ISO codes. "XAB" is used for international documents.
"""

import json

import pandas as pd
from cpr_sdk.parser_models import BaseParserOutput
from rich.console import Console
from rich.progress import track

from scripts.config import interim_data_dir, processed_data_dir, raw_data_dir
from src.geography import geography_string_to_iso
from src.identifiers import Identifier

console = Console()


litigation_us_csv_path = raw_data_dir / "litigation-us.csv"
litigation_non_us_csv_path = raw_data_dir / "litigation-non-us.csv"

litigation_us_df = pd.read_csv(litigation_us_csv_path)
litigation_non_us_df = pd.read_csv(litigation_non_us_csv_path)
litigation_df = pd.concat([litigation_us_df, litigation_non_us_df]).dropna()

litigation_df["Jurisdictions"] = litigation_df["Jurisdictions"].str.split(">").str[0]

# Add an identifier to each document
litigation_df["id"] = litigation_df.apply(
    lambda x: Identifier.generate(x["Title"], x["Document file"]),
    axis=1,
)
litigation_df.set_index("id", inplace=True)


translated_litigation_documents_path = interim_data_dir / "translated" / "litigation"

output_dir = processed_data_dir / "documents" / "litigation"
output_dir.mkdir(parents=True, exist_ok=True)
for document_path in track(
    list(translated_litigation_documents_path.glob("*.json")),
    description="Adding geography ISO to litigation documents",
):
    litigation_document = json.loads(document_path.read_text(encoding="utf-8"))
    parser_output = BaseParserOutput(**litigation_document)

    document_id = document_path.stem

    jurisdiction = litigation_df.loc[document_id, "Jurisdictions"]
    parser_output.document_metadata["geographies"] = [
        geography_string_to_iso(jurisdiction)
    ]

    output_path = output_dir / f"{document_id}.json"
    output_path.write_text(parser_output.model_dump_json())

# Now do the same for the corporate-disclosures dataset, which should be marked as
# international ("XAB"), as the multinational corporations are not limited to a single
# jurisdiction.
corporate_disclosures_docs_dir = (
    interim_data_dir / "translated" / "corporate-disclosures"
)

output_dir = processed_data_dir / "documents" / "corporate-disclosures"
output_dir.mkdir(parents=True, exist_ok=True)

for document_path in track(
    list(corporate_disclosures_docs_dir.glob("*.json")),
    description="Adding geography ISO to corporate disclosure documents",
):
    corporate_disclosure_document = json.loads(
        document_path.read_text(encoding="utf-8")
    )
    parser_output = BaseParserOutput(**corporate_disclosure_document)

    parser_output.document_metadata["geographies"] = ["XAB"]

    output_path = output_dir / document_path.name
    output_path.write_text(parser_output.model_dump_json())
