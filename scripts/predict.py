import tempfile
from pathlib import Path
from typing import Annotated

import boto3
import pandas as pd
import typer
from mypy_boto3_s3.client import S3Client
from rich.console import Console
from rich.progress import track

from scripts.config import aws_region, processed_data_dir
from src.classifier import Classifier
from src.classifier.embedding import EmbeddingClassifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.classifier.stemmed_keyword import StemmedKeywordClassifier
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

console = Console()

app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to run",
            parser=WikibaseID,
        ),
    ],
    save_to_s3: bool = typer.Option(
        False,
        help=(
            "Whether to save the results to S3. "
            "If false, the results will be saved to the local filesystem."
        ),
    ),
    batch_size: int = typer.Option(
        25,
        help="Number of passages to process in each batch",
    ),
):
    """
    Run classifiers on the balanced dataset, and save the results.

    This script runs inference for a set of classifiers on the balanced dataset, and
    saves the resulting positive passages for each concept to a file. The results can
    be saved to S3 or the local filesystem, and used for visualisation (see
    the /predictions_api directory). The predictions_api will read the results from
    S3.

    The script assumes you have already run the `build-dataset` command to create a
    local copy of the balanced dataset.
    """
    if save_to_s3:
        session = boto3.Session(profile_name="labs")
        s3_client: S3Client = session.client("s3", region_name=aws_region)
        bucket_name = "prediction-visualisation"

        # Create the bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except s3_client.exceptions.ClientError:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": aws_region},
            )
            console.log(f"✅ Created S3 bucket: {bucket_name}")
    else:
        s3_client = None
        bucket_name = None

    dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"

    try:
        with console.status("🚚 Loading combined dataset"):
            df = pd.read_feather(dataset_path)
        console.log(f"✅ Loaded {len(df)} passages from {dataset_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{dataset_path} not found locally. If you haven't already, please run:\n"
            "  just build-dataset"
        ) from e

    with console.status("🔍 Fetching concept and subconcepts from Wikibase"):
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(
            wikibase_id, include_labels_from_subconcepts=True
        )

    console.log(f"✅ Fetched {concept} from Wikibase")

    classifiers: list[Classifier] = [
        KeywordClassifier(concept),
        RulesBasedClassifier(concept),
        StemmedKeywordClassifier(concept),
        EmbeddingClassifier(concept, threshold=0.5),
        EmbeddingClassifier(concept, threshold=0.65),
        EmbeddingClassifier(concept, threshold=0.8),
        EmbeddingClassifier(concept, threshold=0.95),
    ]

    for classifier in classifiers:
        classifier.fit()
        console.log(f"✅ Created a {classifier}")
        classifier_path = (
            Path("classifiers") / str(concept.wikibase_id) / f"{classifier.id}.pickle"
        )

        if save_to_s3 and s3_client is not None:
            with tempfile.NamedTemporaryFile() as tmp:
                classifier.save(tmp.name)
                tmp.flush()
                object_name = str(classifier_path).lstrip("/")
                s3_client.upload_file(
                    Filename=tmp.name,
                    Bucket=bucket_name,
                    Key=object_name,
                )
            console.log(f"✅ Saved {classifier} to s3://{bucket_name}/{object_name}")
        else:
            classifier_path = processed_data_dir / classifier_path
            classifier_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.save(classifier_path)
            console.log(f"✅ Saved {classifier} to {classifier_path}")

        labelled_passages: list[LabelledPassage] = []

        n_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
        for batch_start in track(
            range(0, len(df), batch_size),
            console=console,
            transient=True,
            total=n_batches,
            description=f"Running {classifier} on {len(df)} passages in batches of {batch_size}",
        ):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            texts = batch_df["text_block.text"].fillna("").tolist()
            spans_batch = classifier.predict_batch(texts)

            for row, text, spans in zip(batch_df.itertuples(), texts, spans_batch):
                if spans:
                    labelled_passages.append(
                        LabelledPassage(
                            text=text,
                            spans=spans,
                            metadata=row._asdict(),
                        )
                    )

        n_spans = sum(len(entry.spans) for entry in labelled_passages)
        n_positive_passages = sum(len(entry.spans) > 0 for entry in labelled_passages)
        console.log(
            f"✅ Processed {len(df)} passages. Found {n_positive_passages} which mention "
            f'"{classifier.concept}", with {n_spans} individual spans'
        )

        predictions = "\n".join(
            [entry.model_dump_json() for entry in labelled_passages]
        )
        predictions_path = (
            Path("predictions") / str(wikibase_id) / f"{classifier.id}.jsonl"
        )

        if save_to_s3 and s3_client is not None and bucket_name is not None:
            try:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=str(predictions_path).lstrip("/"),
                    Body=predictions,
                )
                console.log(
                    f"✅ Saved predictions to s3://{bucket_name}/{predictions_path}"
                )
            except Exception as e:
                console.log(f"❌ S3 upload failed: {str(e)}")
        else:
            predictions_path = processed_data_dir / predictions_path
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            with open(predictions_path, "w", encoding="utf-8") as f:
                f.write(predictions)
            console.log(f"✅ Saved passages with predictions to {predictions_path}")


if __name__ == "__main__":
    app()
