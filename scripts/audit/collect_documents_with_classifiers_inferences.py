import concurrent.futures
import typer
from pathlib import Path
from typing import Final, Set
from collections.abc import Sequence

from scripts.audit.do_classifier_specs_have_results import collect_file_names
from src.identifiers import WikibaseID
from flows.utils import S3Uri, DocumentImportId, DocumentStem, remove_translated_suffix, Profiler


classifiers: Final[Sequence[str]] = [
    WikibaseID("Q221"),
    WikibaseID("Q639"),
    WikibaseID("Q650"),
    WikibaseID("Q661"),
    WikibaseID("Q777"),
    WikibaseID("Q778"),
]

app = typer.Typer()


@Profiler(printer=print)
def process(classifier: WikibaseID) -> Set[DocumentImportId]:
    print(f"processing {classifier}")
    # Each file name will be something like: s3://cpr-prod-data-pipeline-cache/labelled_passages/Q221/v10/AF.document.002MMUCR.n0000_translated_en.json
    file_names = collect_file_names(
        bucket_name="cpr-prod-data-pipeline-cache",
        prefix=f"labelled_passages/{classifier}/",
        profile_name="production",
    )
    
    # Extract document stems and remove translated suffixes
    document_ids: Set[DocumentImportId] = set()
    for file_name in file_names:
        stem = DocumentStem(Path(file_name).stem)
        document_id = remove_translated_suffix(stem)
        document_ids.add(document_id)
    
    # Write to {classifier}_file_names.txt with one document per line for later use
    output_filename = f"{classifier}_file_names.txt"
    with open(output_filename, "w") as f:
        for document_id in sorted(document_ids):
            f.write(f"{document_id}\n")
    
    print(f"Wrote {len(document_ids)} document IDs to {output_filename}")
    return document_ids


@Profiler(printer=print)
@app.command()
def collect_all() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process, classifier): classifier
            for classifier in classifiers
        }

        all_document_ids: Set[DocumentImportId] = set()

        for future in concurrent.futures.as_completed(futures):
            classifier = futures[future]
            print(f"processed {classifier}")
            try:
                document_ids = future.result()
                all_document_ids.update(document_ids)
            except Exception as exc:
                print(f"{classifier} generated an exception: {str(exc)}")

        # Write the combined set of document IDs to classifiers.txt
        with open("classifiers.txt", "w") as f:
            for document_id in sorted(all_document_ids):
                f.write(f"{document_id}\n")
        
        print(f"Wrote {len(all_document_ids)} unique document IDs to classifiers.txt")


if __name__ == "__main__":
    app()
