"""
Parse a selection of local PDFs, and store the results in S3

This script uses  the `azure_pdf_parser` CLI runner to parse a set of local pdf
documents, and uploads the source folder and results to an AWS S3 bucket for later
retrieval.

The script assumes that the PDFs are stored in the data directory under a structure
like this:

raw/
└── pdfs/
    ├── source1/
    │   ├── abcdef.pdf
    │   └── ...
    ├── source2/
    │   ├── ghijkl.pdf
    │   └── ...
    └── ...

Make sure you've set up your AWS credentials for the labs profile by running
`aws sso login --profile labs` before running this script.

You'll also need a set of environment variables for the Azure API credentials:
AZURE_PROCESSOR_KEY
AZURE_PROCESSOR_ENDPOINT
"""

from azure_pdf_parser.run import run_parser
from dotenv import load_dotenv
from rich.console import Console

from scripts.config import interim_data_dir, raw_data_dir

load_dotenv()

console = Console()

pdf_dir = raw_data_dir / "pdfs"
if not pdf_dir.exists():
    raise FileNotFoundError(
        "The PDFs directory does not exist. "
        "Please create a directory called 'pdfs' in the 'data' directory."
    )
if not any(
    file.suffix == ".pdf"
    for sub_dir in pdf_dir.iterdir()
    if sub_dir.is_dir()
    for file in sub_dir.iterdir()
):
    raise FileNotFoundError(
        "The PDFs directory is empty. "
        "The 'pdfs' directory should contain subdirectories with PDFs to parse."
    )

for pdf_source_directory in pdf_dir.iterdir():
    if not pdf_source_directory.is_dir():
        continue
    console.print(f"📄 Parsing PDFs in {pdf_source_directory.name}")
    output_dir = interim_data_dir / "output" / pdf_source_directory.name
    output_dir.mkdir(exist_ok=True, parents=True)
    run_parser(pdf_dir=pdf_source_directory, output_dir=output_dir)

console.print("📄 All PDFs parsed successfully", style="green")
