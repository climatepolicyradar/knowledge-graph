import os

from typing import Optional
from datasets import Dataset, load_dataset

from src.labelled_passage import LabelledPassage


class HuggingfaceSession:
    """
    A session for interacting with HF

    It solves the storage of LabelledPassages in HF datasets.
    The schema of these datasets is stemming from the LabelledPassage class, and looks as below:
    {
        "id": "string",
        "text": "string",
        "spans": [
            {
                "start_index": 0,
                "end_index": 0,
                "label": "string"
            }
        ],
        "metadata": {}
    }

    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HF_TOKEN")
        self.organisation = "ClimatePolicyRadar"

    def pull(self, dataset_name: str) -> list[LabelledPassage]:
        dataset = load_dataset(
            f"{self.organisation}/{dataset_name}",
            token=self.token,
        )
        labelled_passages = dataset.map(self._labelled_passage_from_row, batched=True).to_list()  # type: ignore
        return labelled_passages

    @staticmethod
    def _labelled_passage_from_row(row: dict) -> LabelledPassage:
        return LabelledPassage(
            id=row["id"],
            text=row["text"],
            spans=row["spans"],
            metadata=row["metadata"],
        )

    def push(self, dataset_name: str, labelled_passages: list[LabelledPassage]) -> None:
        dataset = Dataset.from_dict(
            {
                "id": [lp.id for lp in labelled_passages],
                "text": [lp.text for lp in labelled_passages],
                "spans": [lp.spans for lp in labelled_passages],
                "metadata": [lp.metadata for lp in labelled_passages],
            }
        )
        dataset.push_to_hub(
            repo_id=f"{self.organisation}/{dataset_name}",
            token=self.token,
        )
