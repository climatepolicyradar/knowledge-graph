import os
from typing import Optional

from datasets import Dataset, load_dataset
from huggingface_hub import add_collection_item
from tenacity import retry, stop_after_attempt, wait_fixed

from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span


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
        """Pulls the dataset from the hub"""
        dataset = load_dataset(
            f"{self.organisation}/{dataset_name}",
            token=self.token,
        )

        df = dataset["train"].to_pandas()  # type: ignore
        labelled_passages = list(
            df.apply(lambda x: self._labelled_passage_from_row(x), axis=1)  # type: ignore
        )
        return labelled_passages

    @staticmethod
    def _labelled_passage_from_row(row: dict) -> LabelledPassage:
        return LabelledPassage(
            id=row["id"],
            text=row["text"],
            spans=[Span(**s) for s in row["spans"]],
            metadata=row["metadata"],
        )

    def add_dataset_to_collection(
        self, dataset_name: str, collection_slug: str
    ) -> None:
        """Adds the dataset to a collection"""
        add_collection_item(
            collection_slug=f"{self.organisation}/{collection_slug}",
            item_id=dataset_name,
            exists_ok=True,
            item_type="dataset",
            token=self.token,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def push(self, dataset_name: str, labelled_passages: list[LabelledPassage]) -> None:
        """Pushes the dataset to the hub"""
        dataset = Dataset.from_dict(
            {
                "id": [lp.id for lp in labelled_passages],
                "text": [lp.text for lp in labelled_passages],
                "spans": [
                    [s.model_dump() for s in lp.spans] for lp in labelled_passages
                ],
                "metadata": [lp.metadata for lp in labelled_passages],
            }
        )
        dataset.push_to_hub(
            repo_id=f"{self.organisation}/{dataset_name}",
            token=self.token,
            private=True,
        )
