r"""
Filter BERT output and extract one ``labelled_text`` span per document mention.

    # keep passages with class_1_prob >= threshold, de-duplicate within documents
    uv run python experiments/policy-mentions-classifier/mentions.py filter \
        --bert data/policy-mentions-classifier/fus-609/bert.parquet \
        --run-dir data/policy-mentions-classifier/fus-609

    # estimate LLM cost for the surviving passages
    uv run python experiments/policy-mentions-classifier/mentions.py estimate \
        --run-dir data/policy-mentions-classifier/fus-609

    # label a random sample, then everything
    uv run python experiments/policy-mentions-classifier/mentions.py extract \
        --run-dir data/policy-mentions-classifier/fus-609 --limit 50
    uv run python experiments/policy-mentions-classifier/mentions.py extract \
        --run-dir data/policy-mentions-classifier/fus-609

LLM responses are cached per passage in ``<run-dir>/llm_cache.jsonl``, so re-running
resumes where the last run stopped.
"""

import asyncio
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import Span

console = Console(highlight=False)
app = typer.Typer(no_args_is_help=True)

DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL = "openrouter:openai/gpt-4.1-mini"
OPENROUTER_SSM_PARAM = "/OpenRouter/KGApiKey"

CANDIDATES_FILE = "candidates.parquet"
DUPLICATES_FILE = "duplicates.parquet"
CACHE_FILE = "llm_cache.jsonl"

SYSTEM_PROMPT = """
You are a specialist analyst, tasked with identifying mentions of documents in climate policy texts.
You will mark up references to documents with XML tags.

The concept to identify: a reference to a specific named document which might appear in the
climate policy radar database. That could be a formal citation or a loose, off-hand reference.
These should NOT be the names of governmental bodies or organisations, acronyms, sub-articles,
frameworks, or conventions. Do not label the Paris Agreement, IPCC Guidelines, or simply 'NDC'.
Correct examples: 'Kenya's NDC', 'Singapore's First BTR', 'National Strategy for Environmental Education for Sustainability'
Incorrect examples: 'Article 10', 'UNFCCC', 'Paris', 'COP-28', 'Chapter 3', 'Appendix IX', 'NDC', 'Paris Agreement'

Instructions:
1. Read through the passage carefully.
2. Surround each identified document mention with <concept> tags.
3. If no mentions exist, reproduce the text exactly without any tags.
4. Be very conservative — only label things you are sure are a reference to a specific document.
"""
# Placeholder; "document mention" has no Wikibase concept.
CONCEPT_ID = WikibaseID("Q99999999")


class LLMResponse(BaseModel):
    """Structured LLM output."""

    marked_up_text: str = Field(
        description="The input text reproduced exactly, with <concept> tags around document mentions"
    )
    reasoning: Optional[str] = None  # unused; kept so cache entries share a shape


class MentionExtractor:
    """A pydantic-ai agent plus the identifiers used to key the cache."""

    random_seed = 42
    temperature = 0.0

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        self.agent = Agent(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            output_type=LLMResponse,
            retries=3,
        )
        digest = hashlib.sha256(
            f"{model_name}|{SYSTEM_PROMPT}|{self.temperature}".encode()
        ).hexdigest()[:8]
        self.id = f"mentions-{digest}"

    def __str__(self) -> str:
        """Describe the extractor by model and cache ID."""
        return f"MentionExtractor({self.model_name}, id={self.id})"


def normalise_text(text: str) -> str:
    """Normalise passage text for de-duplication."""
    text = html.unescape(str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- filter


@app.command("filter")
def filter_command(
    bert: Path = typer.Option(..., exists=True, help="Combined output of predict.py."),
    run_dir: Path = typer.Option(..., help="Directory for this run's outputs."),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, help="Min class_1_prob."),
):
    """Keep passages above the BERT threshold and de-duplicate within documents."""
    df = pd.read_parquet(bert) if bert.suffix == ".parquet" else pd.read_csv(bert)
    n_all = len(df)
    all_doc_ids = set(df["document_id"])
    df = df[df["class_1_prob"] >= threshold].copy()
    df = df.sort_values(by=["document_id", "passage_idx"]).reset_index(drop=True)
    df["normalised_text"] = df["text"].map(normalise_text)

    group = df.groupby(["document_id", "normalised_text"], sort=False)
    df["representative_passage_id"] = group["passage_id"].transform("first")
    df["n_duplicates"] = group["passage_id"].transform("size") - 1
    is_rep = df["passage_id"] == df["representative_passage_id"]

    candidates = df[is_rep].drop(columns=["representative_passage_id"])
    duplicates = df[~is_rep].drop(columns=["n_duplicates"])

    run_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(run_dir / CANDIDATES_FILE, index=False)
    duplicates.to_parquet(run_dir / DUPLICATES_FILE, index=False)

    console.print(f"📝 {n_all} passages in BERT output")
    console.print(f"✅ {len(df)} with class_1_prob >= {threshold}")
    console.print(
        f"✅ {len(candidates)} after de-duplication "
        f"({len(duplicates)} duplicates removed)"
    )
    for doc_id in sorted(all_doc_ids - set(df["document_id"])):
        console.print(f"  [yellow]⚠️ no passages above threshold[/]: {doc_id}")
    console.print(f"💾 Saved to [bold green]{run_dir}[/]")


# ------------------------------------------------------------------------- estimate


def _make_classifier(model: str) -> MentionExtractor:
    return MentionExtractor(model)


@app.command()
def estimate(
    run_dir: Path = typer.Option(..., help="Directory containing candidates."),
    model: str = typer.Option(DEFAULT_MODEL, help="pydantic-ai model name."),
    chars_per_token: float = typer.Option(4.0, help="Rough chars-per-token ratio."),
    reasoning_tokens: int = typer.Option(
        100, help="Assumed output tokens per call for the reasoning field + JSON."
    ),
):
    """Estimate token use and cost of the LLM stage from the candidate passages."""
    from knowledge_graph.openrouter_pricing import get_openrouter_pricing

    candidates = pd.read_parquet(run_dir / CANDIDATES_FILE)
    # Building the agent needs a key, but estimating never calls it.
    os.environ.setdefault("OPENROUTER_API_KEY", "dummy-for-estimate")
    system_prompt = _make_classifier(model).system_prompt

    n = len(candidates)
    passage_chars = int(candidates["text"].str.len().sum())
    # Output is the passage echoed back with tags, plus reasoning.
    input_tokens = (n * len(system_prompt) + passage_chars) / chars_per_token
    output_tokens = passage_chars / chars_per_token * 1.05 + n * reasoning_tokens

    console.print(f"📝 {n} passages, {passage_chars:,} characters")
    console.print(f"   system prompt: {len(system_prompt):,} characters per call")
    console.print(f"   ≈ {input_tokens:,.0f} input tokens")
    console.print(f"   ≈ {output_tokens:,.0f} output tokens")

    pricing = asyncio.run(get_openrouter_pricing(model))
    if pricing is None:
        console.print(f"[yellow]Couldn't fetch OpenRouter pricing for {model}[/]")
        return
    cost = (
        input_tokens * pricing.prompt_price + output_tokens * pricing.completion_price
    )
    console.print(
        f"💰 {model}: ${pricing.prompt_price * 1e6:.2f}/M in, "
        f"${pricing.completion_price * 1e6:.2f}/M out"
    )
    console.print(f"💰 Estimated cost ≈ [bold]${cost:,.2f}[/] (excluding retries)")


# -------------------------------------------------------------------------- extract


def _load_cache(path: Path, classifier_id: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry["classifier_id"] == classifier_id:
                    cache[entry["passage_id"]] = entry
    return cache


def _ensure_api_key(model: str, aws_env: AwsEnv) -> None:
    if model.startswith("openrouter:") and not os.environ.get("OPENROUTER_API_KEY"):
        console.print(f"🔑 Fetching OpenRouter API key from SSM ({aws_env})")
        os.environ["OPENROUTER_API_KEY"] = get_aws_ssm_param(
            OPENROUTER_SSM_PARAM, aws_env=aws_env
        )


async def _label_passage(
    classifier: MentionExtractor, passage_id: str, text: str, sem: asyncio.Semaphore
) -> dict:
    """
    Label one passage and return its cache entry.

    ``status`` is "ok", "parse_error" or "api_error". Callers don't cache API
    errors, so they're retried on the next run.
    """
    entry: dict = {
        "passage_id": passage_id,
        "classifier_id": str(classifier.id),
        "model": classifier.model_name,
    }
    async with sem:
        try:
            response = await classifier.agent.run(
                text,
                model_settings=ModelSettings(
                    seed=classifier.random_seed or 42,
                    temperature=classifier.temperature,
                    # without a timeout, a stalled request hangs the run forever
                    timeout=120,
                    # caps runaway repetition; the longest passage needs ~3k tokens
                    max_tokens=8192,
                ),
            )
        except Exception as e:  # network errors, rate limits, retries exhausted
            entry["status"] = "api_error"
            entry["error"] = repr(e)
            return entry

    output = response.output
    if isinstance(output, str):
        output = LLMResponse(marked_up_text=output, reasoning=None)
    entry["marked_up_text"] = output.marked_up_text
    entry["reasoning"] = output.reasoning
    # What the LLM actually tagged, before alignment to the input text.
    entry["raw_spans"] = re.findall(r"<concept>(.*?)</concept>", output.marked_up_text)
    try:
        entry["spans"] = _align_spans(output.marked_up_text, text, str(classifier))
    except Exception as e:
        entry["status"] = "parse_error"
        entry["error"] = repr(e)
        entry["spans"] = []
        return entry

    entry["status"] = "ok"
    return entry


def _align_spans(marked_up_text: str, text: str, labeller: str) -> list[dict]:
    """
    Return the LLM's tagged spans as offsets into the HTML-unescaped passage text.

    The LLM writes HTML entities (``&#39;``, ``&quot;``) back unescaped, so aligning
    against the raw text would shift and clip every later span.
    """
    spans = Span.from_xml(
        xml=marked_up_text,
        concept_id=CONCEPT_ID,
        labellers=[labeller],
        input_text=html.unescape(text),
    )
    return [
        {
            "start_index": s.start_index,
            "end_index": s.end_index,
            "labelled_text": s.labelled_text,
        }
        for s in spans
    ]


async def _run_extraction(
    classifier: MentionExtractor,
    todo: pd.DataFrame,
    cache_path: Path,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    failures: list[dict] = []
    with (
        cache_path.open("a") as cache_file,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Extracting mentions", total=len(todo))
        coros = [
            _label_passage(classifier, str(row["passage_id"]), str(row["text"]), sem)
            for row in todo.to_dict("records")
        ]
        for fut in asyncio.as_completed(coros):
            entry = await fut
            if entry["status"] == "api_error":
                failures.append(entry)
            else:
                # Write as we go, so an interrupted run loses nothing.
                cache_file.write(json.dumps(entry) + "\n")
                cache_file.flush()
            progress.update(task, advance=1)
    return failures


@app.command()
def extract(
    run_dir: Path = typer.Option(..., help="Directory containing candidates."),
    model: str = typer.Option(DEFAULT_MODEL, help="pydantic-ai model name."),
    limit: Optional[int] = typer.Option(
        None, help="Only label a random sample of this many passages."
    ),
    seed: int = typer.Option(42, help="Random seed for --limit sampling."),
    concurrency: int = typer.Option(20, help="Concurrent LLM requests."),
    aws_env: AwsEnv = typer.Option(
        AwsEnv.labs, help="AWS env to fetch the OpenRouter key from, if not set."
    ),
):
    """Extract labelled_text spans from candidate passages with an LLM."""
    _ensure_api_key(model, aws_env)
    classifier = _make_classifier(model)
    console.print(f"🤖 {classifier}")

    candidates = pd.read_parquet(run_dir / CANDIDATES_FILE)
    selected: pd.DataFrame = (
        candidates.sample(n=min(limit, len(candidates)), random_state=seed)
        if limit
        else candidates
    )

    cache_path = run_dir / CACHE_FILE
    cache = _load_cache(cache_path, str(classifier.id))
    todo = pd.DataFrame(selected[~selected["passage_id"].isin(list(cache))])
    console.print(
        f"📝 {len(selected)} passages selected, {len(selected) - len(todo)} cached, "
        f"{len(todo)} to label"
    )

    failures = asyncio.run(_run_extraction(classifier, todo, cache_path, concurrency))
    cache = _load_cache(cache_path, str(classifier.id))

    prefix = f"sample{limit}_" if limit else ""
    spans_df, empty_df, problems_df = _collect(selected, cache, failures)
    spans_df.to_parquet(run_dir / f"{prefix}llm_spans.parquet", index=False)
    empty_df.to_parquet(run_dir / f"{prefix}llm_no_spans.parquet", index=False)
    problems_df.to_parquet(run_dir / f"{prefix}llm_problems.parquet", index=False)

    console.print(
        f"🤓 {len(spans_df)} spans in {spans_df['passage_id'].nunique()} passages; "
        f"{len(empty_df)} passages with no mentions; {len(problems_df)} problems"
    )
    if failures:
        console.print(
            f"[bold red]⚠️ {len(failures)} API failures (not cached). "
            "Re-run the same command to retry them.[/]"
        )
    if limit:
        _print_sample(spans_df)
    console.print(f"💾 Saved to [bold green]{run_dir}[/]")


def _collect(
    selected: pd.DataFrame, cache: dict[str, dict], failures: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Join cached LLM output onto passages.

    Returns (one row per span, passages with no spans, passages that failed).
    """
    span_rows, empty_rows, problem_rows = [], [], []
    failed = {f["passage_id"]: f for f in failures}
    for row in selected.to_dict("records"):
        entry = cache.get(row["passage_id"]) or failed.get(row["passage_id"])
        if entry is None:
            continue
        meta = {
            "llm_status": entry["status"],
            "llm_reasoning": entry.get("reasoning"),
            "llm_raw_spans": json.dumps(entry.get("raw_spans", [])),
        }
        if entry["status"] != "ok":
            problem_rows.append({**row, **meta, "llm_error": entry.get("error")})
            continue
        # Re-align from the cached markup rather than trusting cached offsets.
        row["text"] = html.unescape(row["text"])
        entry["spans"] = _align_spans(
            entry["marked_up_text"], row["text"], entry["classifier_id"]
        )
        if not entry["spans"]:
            empty_rows.append({**row, **meta})
            continue
        raw = set(entry.get("raw_spans", []))
        for span in entry["spans"]:
            span_rows.append(
                {
                    **row,
                    **meta,
                    "labelled_text": span["labelled_text"],
                    "start_index": span["start_index"],
                    "end_index": span["end_index"],
                    # False when the LLM altered the text it tagged.
                    "llm_exact_match": span["labelled_text"] in raw,
                }
            )
    return pd.DataFrame(span_rows), pd.DataFrame(empty_rows), pd.DataFrame(problem_rows)


def _print_sample(spans_df: pd.DataFrame) -> None:
    if spans_df.empty:
        return
    for row in spans_df.head(80).to_dict("records"):
        pages = list(row["page_numbers"])
        page = pages[0] + 1 if pages else "?"  # PAGE_NUMBERS is 0-indexed
        console.print(
            f"[bold cyan]{row['labelled_text']}[/]  "
            f"[dim]({row['document_id']}, p.{page})[/]"
        )


if __name__ == "__main__":
    app()
