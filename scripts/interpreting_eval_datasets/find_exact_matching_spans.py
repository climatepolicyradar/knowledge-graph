from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Typer

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig

console = Console()


app = Typer()


@app.command()
@app.command()
def main(
    config_path: Annotated[Path, Argument(..., help="Path to the sampling config")],
):
    """Loads the labelled passages and filters for exact matches between annotators."""
    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

    for wikibase_id in config.wikibase_ids:
        console.log(f"🔍 Processing {wikibase_id}")
        data_dir = processed_data_dir / "labelled_passages" / wikibase_id
        labelled_passages_path = data_dir / "labelled_passages.jsonl"

        if not labelled_passages_path.exists():
            console.log(
                f"❌ {labelled_passages_path} does not exist. To fetch the data, "
                "run the `save_labelled_passages_from_argilla.py` script using the "
                "same config."
            )
            break

        with open(labelled_passages_path, "r") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]
        n_annotations = sum([len(entry.spans) for entry in labelled_passages])
        console.log(
            f"✅ Loaded {len(labelled_passages)} labelled passages "
            f"with {n_annotations} annotations"
        )

        # produce a golden set by filtering the labelled passages to only include the spans
        # on which annotators agree.
        exact_matching_annotations = []
        for labelled_passage in labelled_passages:
            matching_spans = set()
            for span in labelled_passage.spans:
                for other_span in labelled_passage.spans:
                    if (
                        (
                            span.start_index == other_span.start_index
                            and span.end_index == other_span.end_index
                        )
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        copy_span = span.model_copy()
                        copy_span.labeller = "all"
                        matching_spans.add(copy_span)

            copy_labelled_passage = labelled_passage.model_copy()
            copy_labelled_passage.spans = list(matching_spans)
            exact_matching_annotations.append(copy_labelled_passage)

        n_exact_annotations = sum(
            [len(entry.spans) for entry in exact_matching_annotations]
        )
        console.log(
            f"🤝 Filtered annotations for exact matches. Found {n_exact_annotations}"
        )

        # dump the exact matching annotations to a jsonl file
        exact_matching_annotations_path = data_dir / "exact_matching_annotations.json"
        with open(exact_matching_annotations_path, "w") as f:
            f.writelines([entry.model_dump_json() for entry in labelled_passages])

        console.log(
            f"📝 Wrote {len(exact_matching_annotations)} passages to "
            f"{exact_matching_annotations_path}"
        )


if __name__ == "__main__":
    app()
