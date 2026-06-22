"""
Run the policy-mentions classifier over documents pulled from Snowflake.

Ported from the archived ``policy-mentions-classifier`` repo (``scripts/predict.py``
and ``scripts/use_model.py``). The fine-tuned ModernBERT-base weights live in
``s3://policy-mentions-classifier/models`` and are downloaded into ``./models/``
on first run.

Documents are selected by id or slug; all English passages for each document are
fetched from Snowflake (same connection as ``knowledge_graph.operations.snowflake``),
classified, and written to one CSV per document under ``data/processed/predictions/``.

    uv run python experiments/policy-mentions-classifier/predict.py --document-id <id>
    uv run python experiments/policy-mentions-classifier/predict.py --document-slug <slug>
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knowledge_graph.config import data_dir
from knowledge_graph.operations.snowflake import connect_to_snowflake

console = Console()
app = typer.Typer()

# Model artifacts (kept in S3 so they survive the source repo being archived)
MODEL_S3_BUCKET = "policy-mentions-classifier"
MODEL_S3_PREFIX = "models"
MODEL_DIR = Path(__file__).parent / "models"
BASE_MODEL_ID = "answerdotai/ModernBERT-base"

# Snowflake — schema mirrors knowledge_graph/operations/build_dataset.py
DOCUMENTS_TABLE = "PRODUCTION.PUBLISHED.PIPELINE_DOCUMENTS_V1"
PASSAGES_TABLE = "PRODUCTION.PUBLISHED.PASSAGES_V2"
MINIMUM_TEXT_CHARS = 20
MAX_LENGTH = 512
BATCH_SIZE = 100

# A passage is labelled class 1 (policy mention) only when the model's class-1
# probability meets this threshold; otherwise class 0.
THRESHOLD = 0.75

# Outputs: one CSV per document, named by slug
OUTPUT_DIR = data_dir / "policy-mentions-classifier" / "outputs"


def ensure_model(model_dir: Path = MODEL_DIR) -> None:
    """Download the model from S3 into ``model_dir`` if it isn't already present."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if (model_dir / "model.safetensors").exists():
        console.print(f"Using cached model at [bold white]{model_dir}[/]")
        return

    console.print(
        f"Model not found locally; downloading from "
        f"[bold white]s3://{MODEL_S3_BUCKET}/{MODEL_S3_PREFIX}[/]..."
    )
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MODEL_S3_BUCKET, Prefix=MODEL_S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if obj["Size"] == 0:  # skip the "directory" placeholder key
                continue
            local_path = model_dir / Path(key).name
            console.print(f"  ↓ {key}")
            s3.download_file(MODEL_S3_BUCKET, key, str(local_path))
    console.print("✅ Model downloaded")


def load_model(model_dir: Path = MODEL_DIR):
    """Load the fine-tuned model and tokenizer."""
    console.print(f"Loading trained model from: [bold white]{model_dir}[/]")
    # Tokenizer comes from the original base model (use_fast=False avoids tiktoken issues)
    console.print(f"Loading tokenizer from base model: [bold white]{BASE_MODEL_ID}[/]")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    console.print("✅ Model and tokenizer loaded")

    model_info = {
        "model_name": getattr(model.config, "model_type", "Unknown") + " (fine-tuned)",
        "model_path": str(model_dir),
        "base_model": BASE_MODEL_ID,
    }
    return model, tokenizer, model_info


def load_documents_from_snowflake(
    document_ids: list[str], document_slugs: list[str]
) -> dict:
    """
    Fetch English passages for the requested documents, grouped by document.

    Returns ``{document_id: {document_name, document_slug, passages: [{text,
    text_block_id}]}}`` — the same shape the original predict.py used.
    """
    filters: list[str] = []
    params: list[str] = []
    if document_ids:
        placeholders = ", ".join(["%s"] * len(document_ids))
        filters.append(f"d.DOCUMENT_ID IN ({placeholders})")
        params.extend(document_ids)
    if document_slugs:
        placeholders = ", ".join(["%s"] * len(document_slugs))
        filters.append(f"d.DOCUMENT_SLUG IN ({placeholders})")
        params.extend(document_slugs)
    document_filter = " OR ".join(filters)

    query = f"""
    SELECT
        p.CONTENT AS text_block_text,
        d.DOCUMENT_ID AS document_id,
        d.DOCUMENT_NAME AS document_name,
        d.DOCUMENT_SLUG AS document_slug
    FROM {DOCUMENTS_TABLE} d
    JOIN {PASSAGES_TABLE} p
        ON d.DOCUMENT_ID = p.DOCUMENT_ID
    WHERE p.LANGUAGE = 'en'
      AND p.CONTENT IS NOT NULL
      AND LENGTH(p.CONTENT) > {MINIMUM_TEXT_CHARS}
      AND ({document_filter})
    ORDER BY d.DOCUMENT_ID
    """

    console.print("❄️  Connecting to Snowflake")
    con = connect_to_snowflake(None, None, None)
    cur = con.cursor()
    cur.execute(query, params)
    df = cur.fetch_pandas_all()
    con.close()

    if df.empty:
        return {}

    # Snowflake folds unquoted identifiers to upper case
    df.columns = [c.lower() for c in df.columns]

    document_passage_dict: dict = {}
    for document_id, group in df.groupby("document_id", sort=False):
        # The passages table (per build_dataset.py) exposes no native passage id,
        # so we assign a sequential text_block_id within each document.
        passages = [
            {"text": text, "text_block_id": str(idx)}
            for idx, text in enumerate(group["text_block_text"].tolist())
        ]
        document_passage_dict[document_id] = {
            "document_name": group["document_name"].iloc[0],
            "document_slug": group["document_slug"].iloc[0],
            "passages": passages,
        }

    n_passages = sum(len(d["passages"]) for d in document_passage_dict.values())
    console.print(
        f"📝 Loaded {n_passages} passages from {len(document_passage_dict)} documents"
    )
    return document_passage_dict


def predict_single_text(text: str, model, tokenizer, device: str = "cpu"):
    """Predict a single passage (used as a fallback when a batch fails)."""
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def _result_row(passage: dict, probs) -> dict:
    """Build a result row, applying THRESHOLD to decide predicted_class."""
    class_0_prob, class_1_prob = float(probs[0]), float(probs[1])
    predicted_class = 1 if class_1_prob >= THRESHOLD else 0
    # Report the probability of the class we actually assigned.
    confidence = class_1_prob if predicted_class == 1 else class_0_prob
    return {
        "text": passage["text"],
        "text_block_id": passage["text_block_id"],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_0_prob": class_0_prob,
        "class_1_prob": class_1_prob,
    }


def predict_document(
    passages: list[dict],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
    progress: Optional[Progress] = None,
    task_id=None,
) -> tuple[list[dict], list[dict]]:
    """Classify all passages of a single document. Returns (results, errors)."""
    results: list[dict] = []
    errors: list[dict] = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        texts = [p["text"] for p in batch]
        try:
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, passage in enumerate(batch):
                results.append(_result_row(passage, probabilities[j]))
        except Exception as e:
            console.print(f"Error processing batch: [bold red]{e}[/]")
            # Fall back to one-at-a-time prediction for this batch
            for passage in batch:
                try:
                    _, _, probs = predict_single_text(
                        passage["text"], model, tokenizer, device
                    )
                    results.append(_result_row(passage, probs))
                except Exception as single_e:
                    console.print(
                        f"Error processing single passage: [bold red]{single_e}[/]"
                    )
                    errors.append(
                        {
                            "text": passage["text"],
                            "text_block_id": passage["text_block_id"],
                        }
                    )

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=len(batch))

    return results, errors


OUTPUT_COLUMNS = [
    "text",
    "text_block_id",
    "document_id",
    "document_name",
    "document_slug",
    "predicted_class",
    "confidence",
    "class_0_prob",
    "class_1_prob",
]


def log_prediction_run(log_entries: list[dict], log_file_path: Path) -> None:
    """Append a one-row-per-document summary to a running log CSV."""
    if not log_entries:
        return
    new_df = pd.DataFrame(log_entries)
    if log_file_path.exists():
        try:
            existing = pd.read_csv(log_file_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        except Exception as e:
            console.print(f"⚠️ Could not read existing log ({e}); overwriting")
    new_df.to_csv(log_file_path, index=False)
    console.print(f"📝 Logged run to [bold green]{log_file_path}[/]")


@app.command()
def main(
    document_id: Optional[list[str]] = typer.Option(
        None, "--document-id", help="Document id to classify (repeatable)."
    ),
    document_slug: Optional[list[str]] = typer.Option(
        None, "--document-slug", help="Document slug to classify (repeatable)."
    ),
    batch_size: int = typer.Option(BATCH_SIZE, help="Inference batch size."),
):
    """Classify the passages of the requested documents and save per-document CSVs."""
    document_ids = document_id or []
    document_slugs = document_slug or []
    if not document_ids and not document_slugs:
        console.print(
            "[bold red]Provide at least one --document-id or --document-slug[/]"
        )
        raise typer.Exit(code=1)

    ensure_model()
    model, tokenizer, model_info = load_model()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    console.print(f"Using device: [bold yellow]{device}[/]")

    documents = load_documents_from_snowflake(document_ids, document_slugs)
    if not documents:
        console.print("[bold yellow]No passages found for the requested documents[/]")
        raise typer.Exit()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    current_date = date.today().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_results: list[dict] = []
    log_entries: list[dict] = []
    total_passages = sum(len(d["passages"]) for d in documents.values())

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Labeling passages", total=total_passages)

        for doc_id, doc in documents.items():
            rows, errors = predict_document(
                doc["passages"],
                model,
                tokenizer,
                device,
                batch_size,
                progress=progress,
                task_id=task,
            )

            for r in rows:
                r["document_id"] = doc_id
                r["document_name"] = doc["document_name"]
                r["document_slug"] = doc["document_slug"]

            # Write each document's results immediately so an interrupted run
            # keeps everything completed so far.
            slug = doc["document_slug"] or doc_id
            if rows:
                out_path = output_dir / f"{slug}.csv"
                pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
                console.print(f"✅ Saved {len(rows)} predictions to {out_path}")
            if errors:
                err_path = output_dir / f"{slug}__errors.csv"
                pd.DataFrame(errors).to_csv(err_path, index=False)

            class_1 = sum(1 for r in rows if r["predicted_class"] == 1)
            log_entries.append(
                {
                    "timestamp": timestamp,
                    "date": current_date,
                    "document_id": doc_id,
                    "document_name": doc["document_name"],
                    "model_name": model_info["model_name"],
                    "total_passages": len(rows),
                    "policy_mentions_detected": class_1,
                    "no_policy_mentions": len(rows) - class_1,
                    "errors": len(errors),
                }
            )
            all_results.extend(rows)

    _print_summary(all_results)
    log_prediction_run(log_entries, output_dir.parent / "runs_log.csv")


def _print_summary(results: list[dict]) -> None:
    """Print a rich table + class interpretation across all documents."""
    table = Table(
        title="🎯 Model Predictions",
        show_header=True,
        header_style="bold blue",
        show_lines=True,
    )
    table.add_column("Text", style="white", max_width=60)
    table.add_column("Document", style="white", max_width=40)
    table.add_column("Class", style="cyan", justify="center")
    table.add_column("Confidence", style="green", justify="right")

    # Cap the table at a reasonable number of rows to keep output readable
    for r in results[:50]:
        display_text = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]
        class_color = "green" if r["predicted_class"] == 1 else "red"
        table.add_row(
            display_text,
            r["document_name"],
            f"[{class_color}]{r['predicted_class']}[/]",
            f"{r['confidence']:.3f}",
        )
    console.print(table)
    if len(results) > 50:
        console.print(f"(showing first 50 of {len(results)} passages)")

    class_1 = sum(1 for r in results if r["predicted_class"] == 1)
    avg_conf = float(np.mean([r["confidence"] for r in results])) if results else 0.0
    console.print("\n📊 Summary:")
    console.print(f"  Total predictions: {len(results)}")
    console.print(f"  Class 0 (no mention): {len(results) - class_1}")
    console.print(f"  Class 1 (mention):    {class_1}")
    console.print(f"  Average confidence:   {avg_conf:.3f}")
    console.print(
        "\n💡 Class 0 = [red]no policy mention[/], Class 1 = [green]policy mention[/]"
    )


if __name__ == "__main__":
    app()
