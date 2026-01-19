"""
Script for testing batch prediction using a [SEP] tag.

WARNING: the rest of this script has been claude-coded, with some light touch
supervision. Don't use any of the methods here outside of this script without rereading
and/or rewriting.

Context
-------
The LLMClassifier currently predicts on one text per API call. This script tests
whether combining multiple texts with a [SEP] separator into a single API call
could be faster and cheaper, while measuring any quality degradation.

Potential benefits of batching:
- Fewer API calls = lower latency overhead
- System prompt tokens amortized across multiple texts
- Reduced total cost per text

Potential downsides:
- Longer context may confuse the model
- Output parsing becomes more complex
- Risk of the model losing track of [SEP] boundaries

Output
------
The script produces two analysis tables:

1. Results by Batch Size: Shows speedup vs quality tradeoff for different batch sizes
2. Results by Text Length: Shows whether longer texts degrade more within batches,
   helping separate the effect of batch size from individual text length

Usage
-----
    python scripts/test_batch_prediction_performance.py Q787
    python scripts/test_batch_prediction_performance.py Q787 --batch-sizes 1 2 4 8
    python scripts/test_batch_prediction_performance.py Q787 --model "openrouter:openai/gpt-4o"
    python scripts/test_batch_prediction_performance.py Q787 --output-json results.json

Environment
-----------
Requires OPENROUTER_API_KEY (or appropriate API key for the model provider).
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.table import Table

from knowledge_graph.concept import Concept
from knowledge_graph.config import concept_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import (
    Span,
    SpanXMLConceptFormattingError,
    jaccard_similarity_for_span_lists,
)
from scripts.get_concept import get_concept_async

logger = logging.getLogger(__name__)
console = Console()

# Separator used to join multiple texts in a single prediction
TEXT_SEPARATOR = "\n[SEP]\n"


class BatchLLMResponse(BaseModel):
    """Response format for batched LLM predictions."""

    marked_up_texts: str = Field(
        description=(
            "All input texts reproduced exactly as given, separated by [SEP], "
            "with <concept> tags added where appropriate"
        )
    )
    reasoning: str = Field(
        description="Justification for the concept identifications made"
    )


BATCH_SYSTEM_PROMPT = """
You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
These documents are mostly drawn from a climate and development context.
You will mark up references to concepts with XML tags.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Instructions:

1. You will receive multiple text passages separated by [SEP] markers.
2. Process EACH passage independently - the [SEP] marker indicates a boundary between separate documents.
3. For each passage, identify any mentions of the concept, including references that match the definition.
4. Surround each identified mention with <concept> tags.
5. If a passage contains multiple instances, each one should be tagged separately.
6. If a passage does not contain any instances, reproduce it exactly as given without additional tags.
7. CRITICAL: Preserve the exact [SEP] markers in your output to maintain the document boundaries.
8. Each input text must be reproduced exactly, down to the last character, only adding concept tags.
9. Double check that you have tagged all mentions of the concept.
"""


@dataclass
class PredictionResult:
    """Result of a single or batch prediction."""

    texts: list[str]
    spans_per_text: list[list[Span]]
    elapsed_time_seconds: float
    total_input_chars: int = 0
    total_input_tokens_approx: int = 0


@dataclass
class BatchTestResult:
    """Result of testing a specific batch size."""

    batch_size: int
    total_texts: int
    total_time_seconds: float
    avg_time_per_text_seconds: float
    total_spans_found: int
    total_input_chars: int = 0
    total_input_tokens_approx: int = 0
    avg_chars_per_batch: float = 0.0
    avg_tokens_per_batch_approx: float = 0.0
    jaccard_vs_baseline: Optional[float] = None
    precision_vs_baseline: Optional[float] = None
    recall_vs_baseline: Optional[float] = None
    error_count: int = 0
    predictions: list[PredictionResult] = field(default_factory=list)


@dataclass
class PerTextResult:
    """Result for a single text prediction, used for length analysis."""

    text_index: int
    text_length_chars: int
    text_length_tokens_approx: int
    batch_size: int
    elapsed_time_seconds: float
    num_spans_found: int
    baseline_num_spans: Optional[int] = None
    jaccard_vs_baseline: Optional[float] = None


@dataclass
class LengthBucketResult:
    """Aggregated results for texts in a specific length bucket."""

    bucket_label: str
    num_texts: int
    avg_chars: float
    avg_tokens_approx: float
    avg_time_seconds: float
    total_spans_found: int
    avg_jaccard_vs_baseline: Optional[float] = None


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 characters per token for English)."""
    return len(text) // 4


def load_concept(wikibase_id: WikibaseID) -> Concept:
    """Load a concept from local storage by its Wikibase ID."""
    concept_path = concept_dir / f"{wikibase_id}.json"
    if not concept_path.exists():
        raise FileNotFoundError(
            f"Concept {wikibase_id} not found at {concept_path}. "
            f"Available concepts: {list(concept_dir.glob('*.json'))[:5]}..."
        )
    return Concept.load(concept_path)


def create_batch_agent(
    concept: Concept, model_name: str, random_seed: int = 42
) -> Agent:  # type: ignore[type-arg]
    """Create a pydantic-ai agent for batch predictions."""
    system_prompt = BATCH_SYSTEM_PROMPT.format(
        concept_description=concept.to_markdown()
    )
    return Agent(  # type: ignore[return-value]
        model=model_name,
        system_prompt=system_prompt,
        output_type=BatchLLMResponse,
    )


def parse_batch_response(
    response_text: str,
    original_texts: list[str],
    concept_id: WikibaseID,
    labeller: str,
) -> list[list[Span]]:
    """
    Parse a batch response containing multiple texts separated by [SEP].

    Returns a list of span lists, one per original text.
    """
    # Split by the separator
    response_parts = response_text.split("[SEP]")

    # Clean up whitespace around each part
    response_parts = [part.strip() for part in response_parts]

    # Handle case where response has different number of parts than input
    if len(response_parts) != len(original_texts):
        logger.warning(
            f"Response has {len(response_parts)} parts but expected {len(original_texts)}. "
            f"Attempting best-effort matching."
        )
        # Pad with empty strings if we got fewer parts
        while len(response_parts) < len(original_texts):
            response_parts.append("")

    spans_per_text: list[list[Span]] = []

    for i, (response_part, original_text) in enumerate(
        zip(response_parts[: len(original_texts)], original_texts)
    ):
        try:
            spans = Span.from_xml(
                xml=response_part,
                concept_id=concept_id,
                labellers=[labeller],
                input_text=original_text,
            )
            spans_per_text.append(spans)
        except (SpanXMLConceptFormattingError, Exception) as e:
            logger.warning(f"Failed to parse spans for text {i}: {e}")
            spans_per_text.append([])

    return spans_per_text


def predict_single(
    text: str,
    agent: Agent,  # type: ignore[type-arg]
    concept_id: WikibaseID,
    labeller: str,
    random_seed: int = 42,
) -> PredictionResult:
    """Predict on a single text (baseline approach)."""
    input_chars = len(text)
    input_tokens_approx = estimate_tokens(text)
    start_time = time.perf_counter()

    try:
        response: AgentRunResult[BatchLLMResponse] = agent.run_sync(  # type: ignore[assignment]
            text,
            model_settings=ModelSettings(seed=random_seed),
        )
        elapsed = time.perf_counter() - start_time

        spans = Span.from_xml(
            xml=response.output.marked_up_texts,
            concept_id=concept_id,
            labellers=[labeller],
            input_text=text,
        )

        return PredictionResult(
            texts=[text],
            spans_per_text=[spans],
            elapsed_time_seconds=elapsed,
            total_input_chars=input_chars,
            total_input_tokens_approx=input_tokens_approx,
        )

    except (UnexpectedModelBehavior, SpanXMLConceptFormattingError, Exception) as e:
        elapsed = time.perf_counter() - start_time
        logger.warning(f"Prediction failed: {e}")
        return PredictionResult(
            texts=[text],
            spans_per_text=[[]],
            elapsed_time_seconds=elapsed,
            total_input_chars=input_chars,
            total_input_tokens_approx=input_tokens_approx,
        )


def predict_batch(
    texts: list[str],
    agent: Agent,  # type: ignore[type-arg]
    concept_id: WikibaseID,
    labeller: str,
    random_seed: int = 42,
) -> PredictionResult:
    """Predict on multiple texts joined with [SEP] separator."""
    combined_text = TEXT_SEPARATOR.join(texts)
    input_chars = len(combined_text)
    input_tokens_approx = estimate_tokens(combined_text)

    start_time = time.perf_counter()

    try:
        response: AgentRunResult[BatchLLMResponse] = agent.run_sync(  # type: ignore[assignment]
            combined_text,
            model_settings=ModelSettings(seed=random_seed),
        )
        elapsed = time.perf_counter() - start_time

        spans_per_text = parse_batch_response(
            response_text=response.output.marked_up_texts,
            original_texts=texts,
            concept_id=concept_id,
            labeller=labeller,
        )

        return PredictionResult(
            texts=texts,
            spans_per_text=spans_per_text,
            elapsed_time_seconds=elapsed,
            total_input_chars=input_chars,
            total_input_tokens_approx=input_tokens_approx,
        )

    except (UnexpectedModelBehavior, Exception) as e:
        elapsed = time.perf_counter() - start_time
        logger.warning(f"Batch prediction failed: {e}")
        return PredictionResult(
            texts=texts,
            spans_per_text=[[] for _ in texts],
            elapsed_time_seconds=elapsed,
            total_input_chars=input_chars,
            total_input_tokens_approx=input_tokens_approx,
        )


def calculate_metrics_vs_baseline(
    batch_spans: list[list[Span]],
    baseline_spans: list[list[Span]],
    texts: list[str],
) -> tuple[float, float, float]:
    """
    Calculate Jaccard similarity, precision, and recall vs baseline.

    Returns (jaccard, precision, recall)
    """
    total_jaccard = 0.0
    total_precision_num = 0
    total_precision_denom = 0
    total_recall_num = 0
    total_recall_denom = 0

    for batch_span_list, baseline_span_list, text in zip(
        batch_spans, baseline_spans, texts
    ):
        # Calculate Jaccard similarity for this text
        jaccard = jaccard_similarity_for_span_lists(batch_span_list, baseline_span_list)
        total_jaccard += float(jaccard)

        # Precision: what fraction of batch spans overlap with baseline
        for batch_span in batch_span_list:
            total_precision_denom += 1
            if any(batch_span.overlaps(bs) for bs in baseline_span_list):
                total_precision_num += 1

        # Recall: what fraction of baseline spans are found in batch
        for baseline_span in baseline_span_list:
            total_recall_denom += 1
            if any(baseline_span.overlaps(bs) for bs in batch_span_list):
                total_recall_num += 1

    avg_jaccard = total_jaccard / len(texts) if texts else 0.0
    precision = (
        total_precision_num / total_precision_denom
        if total_precision_denom > 0
        else 1.0
    )
    recall = total_recall_num / total_recall_denom if total_recall_denom > 0 else 1.0

    return avg_jaccard, precision, recall


def run_batch_test(
    texts: list[str],
    batch_size: int,
    agent: Agent,  # type: ignore[type-arg]
    concept_id: WikibaseID,
    labeller: str,
    baseline_spans: Optional[list[list[Span]]] = None,
    random_seed: int = 42,
) -> BatchTestResult:
    """Run predictions with a specific batch size and collect metrics."""
    predictions: list[PredictionResult] = []
    all_spans: list[list[Span]] = []
    error_count = 0
    total_time = 0.0
    total_input_chars = 0
    total_input_tokens_approx = 0

    # Process texts in batches
    num_batches = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        num_batches += 1

        if batch_size == 1:
            result = predict_single(
                text=batch_texts[0],
                agent=agent,
                concept_id=concept_id,
                labeller=labeller,
                random_seed=random_seed,
            )
        else:
            result = predict_batch(
                texts=batch_texts,
                agent=agent,
                concept_id=concept_id,
                labeller=labeller,
                random_seed=random_seed,
            )

        predictions.append(result)
        all_spans.extend(result.spans_per_text)
        total_time += result.elapsed_time_seconds
        total_input_chars += result.total_input_chars
        total_input_tokens_approx += result.total_input_tokens_approx

        # Count errors (empty span lists for texts that should have spans based on baseline)
        if baseline_spans:
            for j, span_list in enumerate(result.spans_per_text):
                text_idx = i + j
                if text_idx < len(baseline_spans):
                    if baseline_spans[text_idx] and not span_list:
                        error_count += 1

    # Calculate metrics vs baseline if provided
    jaccard, precision, recall = None, None, None
    if baseline_spans:
        jaccard, precision, recall = calculate_metrics_vs_baseline(
            batch_spans=all_spans,
            baseline_spans=baseline_spans,
            texts=texts,
        )

    total_spans = sum(len(spans) for spans in all_spans)
    avg_chars_per_batch = total_input_chars / num_batches if num_batches > 0 else 0.0
    avg_tokens_per_batch = (
        total_input_tokens_approx / num_batches if num_batches > 0 else 0.0
    )

    return BatchTestResult(
        batch_size=batch_size,
        total_texts=len(texts),
        total_time_seconds=total_time,
        avg_time_per_text_seconds=total_time / len(texts) if texts else 0,
        total_spans_found=total_spans,
        total_input_chars=total_input_chars,
        total_input_tokens_approx=total_input_tokens_approx,
        avg_chars_per_batch=avg_chars_per_batch,
        avg_tokens_per_batch_approx=avg_tokens_per_batch,
        jaccard_vs_baseline=jaccard,
        precision_vs_baseline=precision,
        recall_vs_baseline=recall,
        error_count=error_count,
        predictions=predictions,
    )


def display_results(results: list[BatchTestResult], baseline_result: BatchTestResult):
    """Display results in a formatted table."""
    table = Table(title="Batch Prediction Performance Results")

    table.add_column("Batch Size", justify="center")
    table.add_column("Avg Chars/Batch", justify="right")
    table.add_column("Avg Tokens/Batch", justify="right")
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Avg Time/Text (s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Spans", justify="right")
    table.add_column("Jaccard", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")

    baseline_time = baseline_result.total_time_seconds

    for result in results:
        speedup = (
            baseline_time / result.total_time_seconds
            if result.total_time_seconds > 0
            else 0
        )

        table.add_row(
            str(result.batch_size),
            f"{result.avg_chars_per_batch:.0f}",
            f"{result.avg_tokens_per_batch_approx:.0f}",
            f"{result.total_time_seconds:.2f}",
            f"{result.avg_time_per_text_seconds:.3f}",
            f"{speedup:.2f}x",
            str(result.total_spans_found),
            f"{result.jaccard_vs_baseline:.3f}"
            if result.jaccard_vs_baseline is not None
            else "N/A",
            f"{result.precision_vs_baseline:.3f}"
            if result.precision_vs_baseline is not None
            else "N/A",
            f"{result.recall_vs_baseline:.3f}"
            if result.recall_vs_baseline is not None
            else "N/A",
        )

    console.print(table)


def extract_per_text_results(
    texts: list[str],
    results: list[BatchTestResult],
    baseline_result: BatchTestResult,
) -> list[PerTextResult]:
    """Extract per-text results from batch results for length analysis."""
    per_text_results: list[PerTextResult] = []

    # Get baseline spans per text
    baseline_spans_per_text: list[list[Span]] = []
    for pred in baseline_result.predictions:
        baseline_spans_per_text.extend(pred.spans_per_text)

    # Process results for each batch size > 1
    for batch_result in results:
        if batch_result.batch_size == 1:
            continue  # Skip baseline

        text_idx = 0
        for pred in batch_result.predictions:
            # Each prediction may contain multiple texts (batch_size texts)
            batch_time_per_text = pred.elapsed_time_seconds / len(pred.texts)

            for i, (text, spans) in enumerate(zip(pred.texts, pred.spans_per_text)):
                if text_idx >= len(texts):
                    break

                baseline_spans = (
                    baseline_spans_per_text[text_idx]
                    if text_idx < len(baseline_spans_per_text)
                    else []
                )

                # Calculate Jaccard for this specific text
                jaccard = float(
                    jaccard_similarity_for_span_lists(spans, baseline_spans)
                )

                per_text_results.append(
                    PerTextResult(
                        text_index=text_idx,
                        text_length_chars=len(text),
                        text_length_tokens_approx=estimate_tokens(text),
                        batch_size=batch_result.batch_size,
                        elapsed_time_seconds=batch_time_per_text,
                        num_spans_found=len(spans),
                        baseline_num_spans=len(baseline_spans),
                        jaccard_vs_baseline=jaccard,
                    )
                )
                text_idx += 1

    return per_text_results


def bucket_by_length(
    per_text_results: list[PerTextResult],
    num_buckets: int = 4,
) -> list[LengthBucketResult]:
    """Bucket per-text results by length and calculate aggregated metrics."""
    if not per_text_results:
        return []

    # Calculate length boundaries from actual results
    text_lengths = [r.text_length_chars for r in per_text_results]
    min_len = min(text_lengths)
    max_len = max(text_lengths)

    if min_len == max_len:
        bucket_boundaries = [(min_len, max_len + 1)]
    else:
        bucket_size = (max_len - min_len) / num_buckets
        bucket_boundaries = [
            (int(min_len + i * bucket_size), int(min_len + (i + 1) * bucket_size))
            for i in range(num_buckets)
        ]
        bucket_boundaries[-1] = (bucket_boundaries[-1][0], max_len + 1)

    bucket_results: list[LengthBucketResult] = []

    for bucket_min, bucket_max in bucket_boundaries:
        bucket_texts = [
            r
            for r in per_text_results
            if bucket_min <= r.text_length_chars < bucket_max
        ]

        if not bucket_texts:
            continue

        avg_chars = sum(r.text_length_chars for r in bucket_texts) / len(bucket_texts)
        avg_tokens = sum(r.text_length_tokens_approx for r in bucket_texts) / len(
            bucket_texts
        )
        avg_time = sum(r.elapsed_time_seconds for r in bucket_texts) / len(bucket_texts)
        total_spans = sum(r.num_spans_found for r in bucket_texts)

        jaccard_values = [
            r.jaccard_vs_baseline
            for r in bucket_texts
            if r.jaccard_vs_baseline is not None
        ]
        avg_jaccard = (
            sum(jaccard_values) / len(jaccard_values) if jaccard_values else None
        )

        bucket_results.append(
            LengthBucketResult(
                bucket_label=f"{bucket_min}-{bucket_max - 1}",
                num_texts=len(bucket_texts),
                avg_chars=avg_chars,
                avg_tokens_approx=avg_tokens,
                avg_time_seconds=avg_time,
                total_spans_found=total_spans,
                avg_jaccard_vs_baseline=avg_jaccard,
            )
        )

    return bucket_results


def display_length_histogram(
    bucket_results: list[LengthBucketResult],
    batch_size: int,
):
    """Display results grouped by text length."""
    table = Table(title=f"Performance by Text Length (Batch Size: {batch_size})")

    table.add_column("Length (chars)", justify="center")
    table.add_column("# Texts", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Spans", justify="right")
    table.add_column("Jaccard", justify="right")

    for bucket in bucket_results:
        table.add_row(
            bucket.bucket_label,
            str(bucket.num_texts),
            f"{bucket.avg_chars:.0f}",
            f"{bucket.avg_tokens_approx:.0f}",
            f"{bucket.avg_time_seconds:.3f}",
            str(bucket.total_spans_found),
            f"{bucket.avg_jaccard_vs_baseline:.3f}"
            if bucket.avg_jaccard_vs_baseline is not None
            else "N/A",
        )

    console.print(table)


def display_all_length_histograms(
    texts: list[str],
    results: list[BatchTestResult],
    baseline_result: BatchTestResult,
    num_buckets: int = 4,
):
    """Display length histograms for each batch size."""
    per_text_results = extract_per_text_results(texts, results, baseline_result)

    if not per_text_results:
        console.print(
            "[yellow]No per-text results available for length analysis.[/yellow]"
        )
        return

    # Group by batch size
    for batch_size in sorted(set(r.batch_size for r in per_text_results)):
        batch_per_text = [r for r in per_text_results if r.batch_size == batch_size]

        if bucket_results := bucket_by_length(batch_per_text, num_buckets):
            display_length_histogram(bucket_results, batch_size)
            console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Test batch prediction performance for LLMClassifier"
    )
    parser.add_argument(
        "wikibase_id",
        type=str,
        help="The Wikibase ID of the concept to test (e.g., Q787)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter:openai/gpt-4o-mini",
        help="Model to use for predictions (default: openrouter:openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=16,
        help="Number of texts to use for testing (default: 16)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save results as JSON",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load concept
    wikibase_id = WikibaseID(args.wikibase_id)

    console.print(f"[bold]Loading concept {wikibase_id}...[/bold]")
    concept = asyncio.run(get_concept_async(wikibase_id))
    console.print(f"  Concept: {concept.preferred_label}")

    if concept.wikibase_id is None:
        console.print("[red]Concept has no wikibase_id set![/red]")
        return

    concept_wikibase_id: WikibaseID = concept.wikibase_id

    # Get sample texts from labelled passages
    labelled_passages = concept.labelled_passages
    if not labelled_passages:
        console.print("[red]No labelled passages found for this concept![/red]")
        return

    # Select texts for testing
    texts = [p.text for p in labelled_passages[: args.num_texts]]
    console.print(f"  Using {len(texts)} texts for testing")

    # Create agent
    console.print(f"[bold]Creating agent with model: {args.model}[/bold]")
    agent = create_batch_agent(
        concept=concept,
        model_name=args.model,
        random_seed=args.random_seed,
    )
    labeller = f"BatchTest({args.model})"

    # Ensure batch_size=1 is included for baseline
    batch_sizes = sorted(set([1] + args.batch_sizes))

    results: list[BatchTestResult] = []
    baseline_result: Optional[BatchTestResult] = None

    for batch_size in batch_sizes:
        console.print(f"\n[bold]Testing batch size: {batch_size}[/bold]")

        result = run_batch_test(
            texts=texts,
            batch_size=batch_size,
            agent=agent,
            concept_id=concept_wikibase_id,
            labeller=labeller,
            baseline_spans=(
                [pred.spans_per_text[0] for pred in baseline_result.predictions]
                if baseline_result and batch_size > 1
                else None
            ),
            random_seed=args.random_seed,
        )

        results.append(result)

        if batch_size == 1:
            baseline_result = result
            console.print(
                f"  [green]Baseline: {result.total_time_seconds:.2f}s, "
                f"{result.total_spans_found} spans found[/green]"
            )
            console.print(
                f"  Avg chars/batch: {result.avg_chars_per_batch:.0f}, "
                f"Avg tokens/batch (approx): {result.avg_tokens_per_batch_approx:.0f}"
            )
        else:
            assert baseline_result is not None  # batch_size=1 always runs first
            speedup = (
                baseline_result.total_time_seconds / result.total_time_seconds
                if result.total_time_seconds > 0
                else 0
            )
            console.print(
                f"  Time: {result.total_time_seconds:.2f}s ({speedup:.2f}x speedup)"
            )
            console.print(
                f"  Avg chars/batch: {result.avg_chars_per_batch:.0f}, "
                f"Avg tokens/batch (approx): {result.avg_tokens_per_batch_approx:.0f}"
            )
            console.print(f"  Spans found: {result.total_spans_found}")
            if result.jaccard_vs_baseline is not None:
                console.print(
                    f"  Jaccard vs baseline: {result.jaccard_vs_baseline:.3f}"
                )
                console.print(
                    f"  Precision: {result.precision_vs_baseline:.3f}, "
                    f"Recall: {result.recall_vs_baseline:.3f}"
                )

    # Display summary tables
    console.print("\n")
    assert baseline_result is not None  # batch_size=1 always runs

    # Table 1: Results by batch size
    console.print("[bold]Table 1: Results by Batch Size[/bold]")
    display_results(results, baseline_result)

    # Table 2: Results by text length (histogram)
    console.print("\n[bold]Table 2: Results by Text Length[/bold]")
    display_all_length_histograms(texts, results, baseline_result, num_buckets=4)

    # Save results as JSON if requested
    if args.output_json:
        output_data = {
            "concept_id": str(wikibase_id),
            "concept_label": concept.preferred_label,
            "model": args.model,
            "num_texts": len(texts),
            "random_seed": args.random_seed,
            "results": [
                {
                    "batch_size": r.batch_size,
                    "total_input_chars": r.total_input_chars,
                    "total_input_tokens_approx": r.total_input_tokens_approx,
                    "avg_chars_per_batch": r.avg_chars_per_batch,
                    "avg_tokens_per_batch_approx": r.avg_tokens_per_batch_approx,
                    "total_time_seconds": r.total_time_seconds,
                    "avg_time_per_text_seconds": r.avg_time_per_text_seconds,
                    "total_spans_found": r.total_spans_found,
                    "jaccard_vs_baseline": r.jaccard_vs_baseline,
                    "precision_vs_baseline": r.precision_vs_baseline,
                    "recall_vs_baseline": r.recall_vs_baseline,
                    "error_count": r.error_count,
                }
                for r in results
            ],
        }

        output_path = Path(args.output_json)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Results saved to {output_path}[/green]")

    # Print conclusions
    console.print("\n[bold]Conclusions:[/bold]")

    if len(results) > 1 and baseline_result is not None:
        baseline_time = baseline_result.total_time_seconds
        best_speedup_result = max(
            [r for r in results if r.batch_size > 1],
            key=lambda r: (baseline_time / r.total_time_seconds)
            if r.total_time_seconds > 0
            else 0.0,
        )
        speedup = baseline_time / best_speedup_result.total_time_seconds

        console.print(
            f"  Best speedup: {speedup:.2f}x at batch size {best_speedup_result.batch_size}"
        )

        if best_speedup_result.jaccard_vs_baseline is not None:
            degradation = 1 - best_speedup_result.jaccard_vs_baseline
            console.print(f"  Quality degradation (1-Jaccard): {degradation:.1%}")

            if degradation < 0.1 and speedup > 1.5:
                console.print(
                    "  [green]Recommendation: Batching appears beneficial with "
                    f"batch size {best_speedup_result.batch_size}[/green]"
                )
            elif degradation >= 0.1:
                console.print(
                    "  [yellow]Warning: Significant quality degradation detected. "
                    "Batching may not be suitable.[/yellow]"
                )
            else:
                console.print(
                    "  [yellow]Note: Minimal speedup. Batching may not be worth the complexity.[/yellow]"
                )


if __name__ == "__main__":
    main()
