"""
Download litigation documents to a local directory

Usage:
poetry run python scripts/sampling_for_sectors_classifier/download_litigation_docs.py
"""

from pathlib import Path

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import track

from src.identifiers import generate_identifier

console = Console()

data_dir = Path("data")

# Read the litigation CSVs
litigation_us_df = pd.read_csv(data_dir / "raw" / "litigation-us.csv")
litigation_non_us_df = pd.read_csv(data_dir / "raw" / "litigation-non-us.csv")

# Assert that the litigation dataframes have the expected columns
expected_columns = ["Title", "Jurisdictions", "Document type", "Document file"]
assert all(
    col in litigation_us_df.columns for col in expected_columns
), "Missing columns in litigation_us_df"
assert all(
    col in litigation_non_us_df.columns for col in expected_columns
), "Missing columns in litigation_non_us_df"

# Join the two dataframes and drop any rows with missing values
litigation_df = pd.concat([litigation_us_df, litigation_non_us_df]).dropna()

# get rid of the extra rubbish in the Jurisdictions column
litigation_df["Jurisdictions"] = litigation_df["Jurisdictions"].str.split(">").str[0]

# Create the PDFs directory
litigation_pdf_dir = data_dir / "raw" / "pdfs" / "litigation"
litigation_pdf_dir.mkdir(exist_ok=True, parents=True)

# Sample from the litigation documents such that we get a maximum volume per
# jurisdiction which is determined by the median number of documents across the 100 most
# common jurisdictions
n_jurisdictions = len(litigation_df["Jurisdictions"].unique())
n_docs_per_jurisdiction = (
    litigation_df.groupby("Jurisdictions").size().sort_values(ascending=False).head(100)
)
# round up to the nearest integer
median_docs_per_jurisdiction = int(n_docs_per_jurisdiction.median().round())
sampled_litigation_df = (
    litigation_df.groupby("Jurisdictions")
    .apply(lambda x: x.sample(min(len(x), median_docs_per_jurisdiction)))
    .reset_index(drop=True)
)


console.print(
    f"📏 Median number of docs per jurisdiction: {median_docs_per_jurisdiction}"
)
console.print(
    f"📄 Sampling {len(sampled_litigation_df)} litigation documents from "
    f"{n_jurisdictions} jurisdictions"
)

# Add an identifier to each document
sampled_litigation_df["id"] = sampled_litigation_df.apply(
    lambda x: generate_identifier(input_string=x["Title"] + x["Document file"]), axis=1
)

# Save the sampled litigation documents as a json file
sampled_litigation_json_path = data_dir / "raw" / "sampled_litigation.json"
sampled_litigation_df.to_json(sampled_litigation_json_path, orient="records")
console.print(
    f"📄 Saved sampled litigation documents to {sampled_litigation_json_path}"
)

# Download the documents
for doc in track(
    sampled_litigation_df.to_dict(orient="records"), description="Downloading documents"
):
    doc_path = litigation_pdf_dir / f"{doc['id']}.pdf"

    if doc_path.exists():
        continue

    response = httpx.get(doc["Document file"])
    with open(doc_path, "wb") as f:
        f.write(response.content)
