"""
Generate evaluation dataset for geography classifier using LLM-assisted annotation.

This script uses pydantic-ai to annotate sample texts with geography mentions,
creating a labelled evaluation set for testing the GeographyClassifier.

Usage:
    uv run python scripts/generate_geography_eval_set.py --help
    uv run python scripts/generate_geography_eval_set.py --input data/sample_texts.txt
    uv run python scripts/generate_geography_eval_set.py --output data/geography_eval_set.json
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Generate geography evaluation dataset using LLM annotation")


class GeographyAnnotation(BaseModel):
    """A single geography mention annotation."""

    span_text: str = Field(description="The exact text that mentions the geography")
    start_index: int = Field(description="Character index where the span starts")
    end_index: int = Field(description="Character index where the span ends")
    wikidata_id: str = Field(
        description="Wikidata QID for the geography (e.g., 'Q155' for Brazil)"
    )
    geography_name: str = Field(description="Human-readable name of the geography")
    confidence: str = Field(
        description="'high', 'medium', or 'low' confidence in this annotation"
    )


class AnnotatedText(BaseModel):
    """A fully annotated text with geography mentions."""

    text: str = Field(description="The original input text, unchanged")
    annotations: list[GeographyAnnotation] = Field(
        description=(
            "List of geography mentions found in the text. Empty list if none found."
        )
    )
    reasoning: str = Field(description="Brief explanation of the annotation decisions")


ANNOTATION_SYSTEM_PROMPT = """
You are an expert annotator for geography named entity recognition.

Your task is to identify all explicit mentions of countries, nations, cities, regions,
and other geographic locations in the given text.

Guidelines:
1. Only annotate EXPLICIT mentions - the text must name or directly reference the geography
2. Include demonyms (e.g., "Brazilian" → Brazil/Q155, "French" → France/Q142)
3. Include abbreviations (e.g., "UK" → United Kingdom/Q145, "US" → United States/Q30)
4. Do NOT annotate:
   - Organizations named after places (e.g., "Oxford University" is NOT Oxford the city)
   - Generic geographic terms without specific referents (e.g., "the country" without context)
5. Use standard Wikidata QIDs for geographies
6. Provide start_index and end_index as character offsets (0-indexed)
7. If unsure, mark confidence as 'low'

Common Wikidata QIDs for reference:
- Brazil: Q155, United Kingdom: Q145, United States: Q30, France: Q142
- China: Q148, India: Q668, Germany: Q183, Japan: Q17
- Georgia (country): Q230, Georgia (US state): Q1428
- Australia: Q408, Canada: Q16, Mexico: Q96, Spain: Q29
- Italy: Q38, Netherlands: Q55, Poland: Q36, Russia: Q159
- South Africa: Q258, Nigeria: Q1033, Kenya: Q114, Egypt: Q79
- Argentina: Q414, Chile: Q298, Colombia: Q739, Peru: Q419

For ambiguous cases like "Georgia", use context to determine which entity is meant.
If truly ambiguous, annotate with confidence='low'.

IMPORTANT: Double-check that start_index and end_index correctly bound the span_text
within the original text. The text[start_index:end_index] should exactly match span_text.
"""


async def annotate_texts(
    texts: Sequence[str],
    model_name: str = "anthropic:claude-sonnet-4-20250514",
    batch_size: int = 10,
) -> list[AnnotatedText]:
    """
    Annotate a batch of texts with geography mentions using an LLM.

    Args:
        texts: Texts to annotate.
        model_name: pydantic-ai model identifier.
        batch_size: Number of concurrent requests.

    Returns:
        List of AnnotatedText objects.
    """
    agent: Agent[None, AnnotatedText] = Agent(
        model_name,
        system_prompt=ANNOTATION_SYSTEM_PROMPT,
        output_type=AnnotatedText,
    )

    async def annotate_one(text: str) -> AnnotatedText:
        result = await agent.run(
            f"Annotate all geography mentions in this text:\n\n{text}",
            model_settings=ModelSettings(temperature=0.0),
        )
        return result.output

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = await asyncio.gather(*[annotate_one(t) for t in batch])
        results.extend(batch_results)
        logger.info(f"Annotated {len(results)}/{len(texts)} texts")

    return results


def validate_annotations(annotations: list[AnnotatedText]) -> list[dict]:
    """
    Validate annotations and flag issues for human review.

    Args:
        annotations: List of annotated texts to validate.

    Returns:
        List of issues found, each as a dict with keys:
        - text_idx: Index of the text in annotations list
        - annotation_idx: Index of the annotation within the text
        - issue: Type of issue found
        - details: Human-readable description of the issue
    """
    issues = []
    for i, ann in enumerate(annotations):
        for j, geo in enumerate(ann.annotations):
            # Check span bounds
            if geo.start_index < 0 or geo.end_index > len(ann.text):
                issues.append(
                    {
                        "text_idx": i,
                        "annotation_idx": j,
                        "issue": "span_out_of_bounds",
                        "details": (
                            f"[{geo.start_index}:{geo.end_index}] "
                            f"vs text len {len(ann.text)}"
                        ),
                    }
                )
                continue

            # Check span text matches
            actual_span = ann.text[geo.start_index : geo.end_index]
            if actual_span.lower() != geo.span_text.lower():
                issues.append(
                    {
                        "text_idx": i,
                        "annotation_idx": j,
                        "issue": "span_text_mismatch",
                        "details": f"Expected '{geo.span_text}', got '{actual_span}'",
                    }
                )

            # Flag low confidence for review
            if geo.confidence == "low":
                issues.append(
                    {
                        "text_idx": i,
                        "annotation_idx": j,
                        "issue": "low_confidence",
                        "details": geo.span_text,
                    }
                )

            # Validate Wikidata ID format
            if not geo.wikidata_id.startswith("Q") or not geo.wikidata_id[1:].isdigit():
                issues.append(
                    {
                        "text_idx": i,
                        "annotation_idx": j,
                        "issue": "invalid_wikidata_id",
                        "details": f"'{geo.wikidata_id}' is not a valid QID format",
                    }
                )

    return issues


# Sample texts for testing - these can be replaced with actual corpus samples
SAMPLE_TEXTS = [
    "The Brazilian government announced new climate policies today.",
    "Representatives from the UK and France met in Geneva.",
    "Oxford University researchers published findings on Amazon deforestation.",
    "The Georgian wine industry has grown significantly since EU trade agreements.",
    "Climate change affects coastal cities like Miami, Mumbai, and Jakarta.",
    "The Amazon rainforest spans Brazil, Peru, Colombia, and several other countries.",
    "German automotive companies are investing heavily in electric vehicles.",
    "The Nile River flows through multiple African nations including Egypt and Sudan.",
    "Chinese and American negotiators discussed trade relations in Washington.",
    "Australian bushfires have increased in frequency due to global warming.",
    "The Paris Agreement was signed by nearly 200 countries in 2015.",
    "South African mining companies face new environmental regulations.",
    "Japanese technology firms are expanding operations in Southeast Asia.",
    "Canadian oil sands development remains controversial among environmentalists.",
    "The European Union proposed new carbon tariffs affecting imports from Asia.",
    "Mexican authorities announced water conservation measures for Mexico City.",
    "Indian farmers protested new agricultural policies in New Delhi.",
    "Norwegian sovereign wealth fund divested from fossil fuel companies.",
    "The Congo Basin contains the world's second-largest rainforest.",
    "Russian gas exports to Europe have declined significantly.",
]


@app.command()
def generate(
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to file with texts to annotate (one per line)",
    ),
    output_file: Path = typer.Option(
        Path("data/geography_eval_set.json"),
        "--output",
        "-o",
        help="Path to save the evaluation dataset",
    ),
    model: str = typer.Option(
        "anthropic:claude-sonnet-4-20250514",
        "--model",
        "-m",
        help="pydantic-ai model identifier",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        "-b",
        help="Number of concurrent annotation requests",
    ),
    use_samples: bool = typer.Option(
        False,
        "--use-samples",
        help="Use built-in sample texts instead of input file",
    ),
) -> None:
    """Generate geography evaluation dataset using LLM annotation."""
    # Load texts
    if use_samples:
        texts = SAMPLE_TEXTS
        logger.info(f"Using {len(texts)} built-in sample texts")
    elif input_file:
        if not input_file.exists():
            raise typer.BadParameter(f"Input file not found: {input_file}")
        texts = [
            line.strip() for line in input_file.read_text().splitlines() if line.strip()
        ]
        logger.info(f"Loaded {len(texts)} texts from {input_file}")
    else:
        raise typer.BadParameter("Either --input or --use-samples must be specified")

    # Annotate
    logger.info(f"Annotating {len(texts)} texts with model {model}...")
    annotations = asyncio.run(
        annotate_texts(texts, model_name=model, batch_size=batch_size)
    )

    # Validate
    issues = validate_annotations(annotations)
    logger.info(f"Found {len(issues)} issues to review")

    # Prepare output
    output = {
        "annotations": [a.model_dump() for a in annotations],
        "issues": issues,
        "metadata": {
            "total_texts": len(texts),
            "total_annotations": sum(len(a.annotations) for a in annotations),
            "model": model,
        },
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved evaluation dataset to {output_file}")

    # Print summary
    typer.echo("\nSummary:")
    typer.echo(f"  Total texts: {len(texts)}")
    typer.echo(f"  Total annotations: {output['metadata']['total_annotations']}")
    typer.echo(f"  Issues to review: {len(issues)}")

    if issues:
        typer.echo("\nIssue breakdown:")
        issue_types = {}
        for issue in issues:
            issue_types[issue["issue"]] = issue_types.get(issue["issue"], 0) + 1
        for issue_type, count in sorted(issue_types.items()):
            typer.echo(f"  {issue_type}: {count}")


@app.command()
def validate(
    input_file: Path = typer.Argument(
        ...,
        help="Path to evaluation dataset JSON file",
    ),
) -> None:
    """Validate an existing evaluation dataset."""
    if not input_file.exists():
        raise typer.BadParameter(f"File not found: {input_file}")

    data = json.loads(input_file.read_text())
    annotations = [AnnotatedText.model_validate(a) for a in data["annotations"]]

    issues = validate_annotations(annotations)

    typer.echo(f"Validation results for {input_file}:")
    typer.echo(f"  Total texts: {len(annotations)}")
    typer.echo(f"  Total annotations: {sum(len(a.annotations) for a in annotations)}")
    typer.echo(f"  Issues found: {len(issues)}")

    if issues:
        typer.echo("\nIssue breakdown:")
        issue_types = {}
        for issue in issues:
            issue_types[issue["issue"]] = issue_types.get(issue["issue"], 0) + 1
        for issue_type, count in sorted(issue_types.items()):
            typer.echo(f"  {issue_type}: {count}")


if __name__ == "__main__":
    app()
