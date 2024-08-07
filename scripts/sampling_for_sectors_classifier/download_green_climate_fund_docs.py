"""
Download Green Climate Fund documents to a local directory

Use the output of the scraper to download the Green Climate Fund documents from the
website and save them to a local data directory. Assumes you have the output of the
scraper in a file called `green-climate-fund.json`.

Usage:
    poetry run python scripts/sampling_for_sectors_classifier/download_green_climate_fund_docs.py
"""

import json
import sys

import httpx
from rich.console import Console
from rich.progress import track

from scripts.config import raw_data_dir

console = Console()

scraped_json_path = raw_data_dir / "green-climate-fund.json"
if not scraped_json_path.exists():
    console.print(
        "🚨 The scraped JSON file does not exist. Please run the scraper first.",
        style="bold red",
    )
    sys.exit(1)

with open(scraped_json_path, "r") as f:
    scraped_data = json.load(f)

# Create the PDFs directory
gcf_pdf_dir = raw_data_dir / "pdfs" / "green-climate-fund"
gcf_pdf_dir.mkdir(exist_ok=True, parents=True)

# Download the documents
for doc in track(scraped_data, description="Downloading documents"):
    doc_path = gcf_pdf_dir / f"{doc['id']}.pdf"

    if doc_path.exists():
        continue

    response = httpx.get(doc["pdf_url"])
    with open(doc_path, "wb") as f:
        f.write(response.content)
