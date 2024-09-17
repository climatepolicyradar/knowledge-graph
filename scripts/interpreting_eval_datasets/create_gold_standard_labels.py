from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Option, Typer

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig
from src.span import merge_overlapping_spans

console = Console(highlight=False)


app = Typer()


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
):
    """
    Create a gold standard dataset using data from multiple human annotators

    The script uses a supplied configuration file to determine which datasets to process.

    Before running this script, you should have run the `save_labelled_passages_from_argilla.py`
    """

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

    for wikibase_id in config.wikibase_ids:
        console.log(f"🤔 Processing {wikibase_id}")
        data_dir = processed_data_dir / "labelled_passages" / wikibase_id
        labelled_passages_path = data_dir / "labelled_passages.jsonl"

        if not labelled_passages_path.exists():
            console.log(
                f"❌ {labelled_passages_path} does not exist. To fetch the data, "
                "run the `save_labelled_passages_from_argilla.py` script using the "
                "same config."
            )
            break

        with open(labelled_passages_path, "r", encoding="utf-8") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]

        gold_standard_passages: list[LabelledPassage] = []

        for labelled_passage in labelled_passages:
            merged_spans = merge_overlapping_spans(
                # if there's any overlap between spans, merge them
                spans=labelled_passage.spans,
                jaccard_threshold=0,
            )
            gold_standard_passages.append(
                labelled_passage.model_copy(update={"spans": merged_spans}, deep=True)
            )
        n_annotations = sum([len(entry.spans) for entry in gold_standard_passages])
        console.log(
            f"🚚 Loaded {len(gold_standard_passages)} labelled passages "
            f"with {n_annotations} individual annotations"
        )

        gold_standard_path = data_dir / "gold_standard.jsonl"
        with open(gold_standard_path, "w", encoding="utf-8") as f:
            f.writelines(
                [entry.model_dump_json() + "\n" for entry in gold_standard_passages]
            )

        console.log(f"✅ Saved gold standard passages to {gold_standard_path}")


if __name__ == "__main__":
    app()
