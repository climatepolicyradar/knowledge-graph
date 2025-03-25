"""
Download litigation documents to a local directory.

This script loads and combines US and non-US litigation documents from CSV files stored
in the `data/raw` directory. The dataset is heavily geographically imbalanced, so the
script samples the dataset to ensure balanced representation of jurisdictions before
downloading the documents.

The sampling strategy:
- Takes the median number of documents from the top 100 jurisdictions
- Applies this as a cap for all jurisdictions
- Ensures no jurisdiction is over-represented in the final dataset

Input files:
- data/raw/litigation-us.csv: US litigation documents
- data/raw/litigation-non-us.csv: Non-US litigation documents

Output files:
- data/raw/sampled_litigation.json: Metadata for sampled documents
- data/raw/pdfs/litigation/*.pdf: Downloaded PDF documents
"""

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import track

from scripts.config import raw_data_dir
from src.identifiers import generate_identifier

console = Console()


# Read the litigation CSVs
litigation_us_df = pd.read_csv(raw_data_dir / "litigation-us.csv")
litigation_non_us_df = pd.read_csv(raw_data_dir / "litigation-non-us.csv")

# Assert that the litigation dataframes have the expected columns
expected_columns = ["Title", "Jurisdictions", "Document type", "Document file"]
assert all(col in litigation_us_df.columns for col in expected_columns), (
    "Missing columns in litigation_us_df"
)
assert all(col in litigation_non_us_df.columns for col in expected_columns), (
    "Missing columns in litigation_non_us_df"
)

# Join the two dataframes and drop any rows with missing values
litigation_df = pd.concat([litigation_us_df, litigation_non_us_df]).dropna()

# get rid of the extra rubbish in the Jurisdictions column
litigation_df["Jurisdictions"] = litigation_df["Jurisdictions"].str.split(">").str[0]

# Create the PDFs directory
litigation_pdf_dir = raw_data_dir / "pdfs" / "litigation"
litigation_pdf_dir.mkdir(exist_ok=True, parents=True)

# Sample from the litigation documents such that we get a maximum volume per
# jurisdiction which is determined by the median number of documents across the 100 most
# common jurisdictions
n_jurisdictions = len(litigation_df["Jurisdictions"].unique())
n_docs_per_jurisdiction = (
    litigation_df.groupby("Jurisdictions").size().sort_values(ascending=False).head(100)  # type: ignore
)
# round up to the nearest integer
median_docs_per_jurisdiction = int(n_docs_per_jurisdiction.median().round())  # type: ignore
sampled_litigation_df = (
    litigation_df.groupby("Jurisdictions")[["Title", "Document file", "Jurisdictions"]]
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
    lambda x: generate_identifier(x["Title"], x["Document file"]), axis=1
)

# Save the litigation documents as a json file
sampled_litigation_json_path = raw_data_dir / "sampled_litigation.json"
sampled_litigation_df.to_json(sampled_litigation_json_path, orient="records")
console.print(f"📄 Saved litigation documents to {sampled_litigation_json_path}")

# Download the documents
for doc in track(
    sampled_litigation_df.to_dict(orient="records"),  # type: ignore
    description="Downloading documents",
):
    doc_path = litigation_pdf_dir / f"{doc['id']}.pdf"

    if doc_path.exists():
        continue

    response = httpx.get(doc["Document file"])
    with open(doc_path, "wb") as f:
        f.write(response.content)
