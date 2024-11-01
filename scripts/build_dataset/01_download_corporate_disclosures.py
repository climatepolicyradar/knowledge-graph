"""
Prompt the user to download the corporate-disclosures documents from the Google Drive link.

The downloaded files should be saved to the `corporate_disclosures_pdf_dir` directory.

The script can't download the files directly from the Google Drive link because the
link is not publicly accessible. We assume that the user is a member of the CPR team
with access to the Google Drive folder.
"""

from rich.console import Console

from scripts.config import raw_data_dir

console = Console()

corporate_disclosures_pdf_dir = raw_data_dir / "pdfs" / "corporate-disclosures"
corporate_disclosures_pdf_dir.mkdir(parents=True, exist_ok=True)

# check whether there are pdf files in the directory
if len(list(corporate_disclosures_pdf_dir.glob("*.pdf"))) > 0:
    console.print(
        f"✅ PDF documents are already saved in {corporate_disclosures_pdf_dir}."
    )
    exit()


drive_url = "https://drive.google.com/drive/folders/1OR8-u4C4VDsanQgtLOCZBbqi2QK_TNGR?usp=drive_link"

console.print(
    f"📥 Please download the corporate disclosure documents from the following link: {drive_url}"
)
console.print(
    f"📂 The downloaded files should be saved to {corporate_disclosures_pdf_dir}"
)
