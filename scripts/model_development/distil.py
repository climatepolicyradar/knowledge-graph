import os
import random

import boto3
import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from knowledge_graph.classifier import (
    BertBasedClassifier,
    EmbeddingClassifier,
    LLMClassifier,
)
from knowledge_graph.config import equity_columns, processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import count_passage_level_metrics
from knowledge_graph.sampling import create_balanced_sample
from knowledge_graph.wikibase import WikibaseSession
from scripts.model_development.local_inference_utilities import (
    label_passages,
    save_labelled_passages_and_classifier,
)

app = typer.Typer()
console = Console(highlight=False)


@app.command()
def main(
    concept_id: str = typer.Argument(...),
    embedding_classifier_sample_size: int = typer.Option(
        50_000,
        help="The number of passages to predict against with the embedding classifier",
    ),
    embedding_classifier_threshold: float = typer.Option(
        0.6,
        help=(
            "The threshold for the embedding classifier. Makes sense to set this a "
            "little bit lower than the default threshold of 0.65, to give the LLM "
            "classifier a chance to correct any false positives which would otherwise "
            "be left out of the set for fine-tuning."
        ),
    ),
    embedding_classifier_batch_size: int = typer.Option(
        500, help="The batch size for making predictions with the embedding classifier"
    ),
    llm_classifier_sample_size: int = typer.Option(
        2_500, help="The number of passages to predict against with the LLM classifier"
    ),
    llm_classifier_batch_size: int = typer.Option(
        50, help="The batch size for making predictions with the LLM classifier"
    ),
    bert_based_classifier_batch_size: int = typer.Option(
        100, help="The batch size for making predictions with the BERT-based classifier"
    ),
    llm_model_name: str = typer.Option(
        "gemini-1.5-flash",
        help=(
            "The name of the model to use. See https://ai.pydantic.dev/models/ for a "
            "list of available models and the necessary environment variables needed "
            "to run each."
        ),
    ),
):
    """
    Distils a BERT-based classifier from an LLM-labelled dataset.

    This script implements a multi-stage process to create a BERT-based classifier for
    a given concept:
    1. Uses an embedding classifier to quickly filter a large dataset
    2. Applies an LLM classifier to refine those results, resulting in a set of
       labelled passages
    3. Fine-tunes a BERT-based classifier on the LLM-labelled passages
    4. Evaluates the final classifier's performance against the "ground truth" as given
       by the LLM classifier

    Note:
        Requires AWS SSM parameter store access for the Google API key
        Intermediate results and the final classifier are saved to standardised paths
    """
    wikibase = WikibaseSession()

    concept = wikibase.get_concept(WikibaseID(concept_id))
    console.log(f"🧠 Loaded concept: [bold white]{concept}[/bold white]")

    with console.status("Loading the combined dataset..."):
        combined_df = pd.read_feather(processed_data_dir / "combined_dataset.feather")
    console.log(f"✅ Combined dataset loaded with {len(combined_df)} rows")

    with console.status(
        "🧪 Sampling a balanced set of passages from the combined dataset"
    ):
        balanced_sample_dataframe: pd.DataFrame = create_balanced_sample(
            df=combined_df,
            sample_size=embedding_classifier_sample_size,
            on_columns=equity_columns,
        )

        passages_to_label_with_embedding_classifier: list[LabelledPassage] = [
            LabelledPassage(text=str(row["text_block.text"]), spans=[])
            for _, row in balanced_sample_dataframe.iterrows()
        ]
    console.log(
        f"✅ Sampled {len(passages_to_label_with_embedding_classifier)} passages from the combined dataset"
    )

    embedding_classifier = EmbeddingClassifier(
        concept, threshold=embedding_classifier_threshold
    )
    console.log(f"🤖 Created an {embedding_classifier}")

    passages_labelled_by_embedding_classifier = label_passages(
        labelled_passages=passages_to_label_with_embedding_classifier,
        classifier=embedding_classifier,
        batch_size=embedding_classifier_batch_size,
    )

    positive_passages_from_embedding_classifier = [
        passage
        for passage in passages_labelled_by_embedding_classifier
        if passage.spans
    ]
    percentage_positive = int(
        len(positive_passages_from_embedding_classifier)
        / len(passages_labelled_by_embedding_classifier)
        * 100
    )
    console.log(
        f"✅ Labelled {len(passages_labelled_by_embedding_classifier)} passages with "
        f"{embedding_classifier.name}. {percentage_positive}% of passages were positive"
    )

    save_labelled_passages_and_classifier(
        labelled_passages=passages_labelled_by_embedding_classifier,
        classifier=embedding_classifier,
    )

    ssm = boto3.client("ssm", region_name="eu-west-1")
    response = ssm.get_parameter(Name="/RAG/GOOGLE_API_KEY", WithDecryption=True)
    os.environ["GEMINI_API_KEY"] = response["Parameter"]["Value"]
    llm_classifier = LLMClassifier(concept, model_name=llm_model_name)
    console.log(f"🤖 Created a {llm_classifier}")

    if len(positive_passages_from_embedding_classifier) > llm_classifier_sample_size:
        console.log(
            f"🧪 Sampling {llm_classifier_sample_size} passages from the "
            f"{len(positive_passages_from_embedding_classifier)} passages which seem "
            "positive using the embedding classifier"
        )
        positive_passages_from_embedding_classifier = random.sample(
            positive_passages_from_embedding_classifier, llm_classifier_sample_size
        )

    passages_labelled_by_llm_classifier = label_passages(
        labelled_passages=positive_passages_from_embedding_classifier,
        classifier=llm_classifier,
        batch_size=llm_classifier_batch_size,
    )

    positive_passages_from_llm_classifier = [
        passage for passage in passages_labelled_by_llm_classifier if passage.spans
    ]
    percentage_positive = int(
        len(positive_passages_from_llm_classifier)
        / len(passages_labelled_by_llm_classifier)
        * 100
    )
    console.log(
        f"✅ Labelled {len(passages_labelled_by_llm_classifier)} passages with "
        f"{llm_classifier.name}. {percentage_positive}% of passages were positive"
    )

    save_labelled_passages_and_classifier(
        labelled_passages=passages_labelled_by_llm_classifier,
        classifier=llm_classifier,
    )

    concept.labelled_passages = passages_labelled_by_llm_classifier

    bert_based_classifier = BertBasedClassifier(concept=concept)
    console.log(f"🤖 Created a {bert_based_classifier}")

    console.log(
        "🪓 Creating a train/test split of the LLM-labelled passages...",
    )
    train_passages, test_passages = train_test_split(
        passages_labelled_by_llm_classifier, test_size=0.2, random_state=42
    )

    with console.status(
        f"🔧 Fine-tuning the {bert_based_classifier.name} based on LLM predictions..."
    ):
        bert_based_classifier.fit(labelled_passages=train_passages, use_wandb=False)
    console.log("🔧 Fine-tuning completed.")

    passages_labelled_by_bert_based_classifier = label_passages(
        labelled_passages=test_passages,
        classifier=bert_based_classifier,
        batch_size=bert_based_classifier_batch_size,
    )

    positive_passages_from_bert_based_classifier = [
        passage
        for passage in passages_labelled_by_bert_based_classifier
        if passage.spans
    ]
    percentage_positive = int(
        len(positive_passages_from_bert_based_classifier)
        / len(passages_labelled_by_bert_based_classifier)
        * 100
    )
    console.log(
        f"✅ Labelled {len(passages_labelled_by_bert_based_classifier)} passages with "
        f"{bert_based_classifier.name}. {percentage_positive}% of passages were positive"
    )

    save_labelled_passages_and_classifier(
        labelled_passages=passages_labelled_by_bert_based_classifier,
        classifier=bert_based_classifier,
    )

    console.log(
        "👨‍🔬 Calculating passage-level performance metrics, based on ground truth as given by the LLM classifier..."
    )
    cm = count_passage_level_metrics(
        ground_truth_passages=test_passages,
        predicted_passages=passages_labelled_by_bert_based_classifier,
    )

    metrics_table = Table(box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Score", style="magenta")
    metrics_table.add_row("Precision", f"{cm.precision():.3f}")
    metrics_table.add_row("Recall", f"{cm.recall():.3f}")
    metrics_table.add_row("F1 Score", f"{cm.f1_score():.3f}")
    metrics_table.add_row("Support (Passages)", f"{cm.support()}")
    console.log(metrics_table)

    console.log("Sample of final positive passages:", style="bold white")
    for passage in random.sample(
        [
            passage
            for passage in passages_labelled_by_bert_based_classifier
            if passage.spans
        ],
        10,
    ):
        console.log(passage.text, highlight=False, end="\n\n")

    console.log("Sample of final negative passages:", style="bold white")
    for passage in random.sample(
        [
            passage
            for passage in passages_labelled_by_bert_based_classifier
            if not passage.spans
        ],
        10,
    ):
        console.log(passage.text, highlight=False, end="\n\n")


if __name__ == "__main__":
    app()
