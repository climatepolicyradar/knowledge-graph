r"""
Flag excluded mentions and write the mentions workbook.

    uv run --with openpyxl python experiments/policy-mentions-classifier/deliverable.py \
        --run-dir data/policy-mentions-classifier/fus-609 \
        --bert data/policy-mentions-classifier/fus-609/bert.parquet \
        --ids-file experiments/policy-mentions-classifier/inputs/fus-609.txt

Writes ``BTR_NDC_mentions_<YYYY-MM-DD>.xlsx`` to the run dir.

Flagged rows are moved to the "excluded mentions" tab, never dropped.
"""

import html
import json
import re
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import httpx
import pandas as pd
import typer
from rich.console import Console

console = Console(highlight=False)

SEARCH_URL = "https://app.climatepolicyradar.org/search"
GEOGRAPHIES_URL = "https://api.climatepolicyradar.org/geographies/"

# Mirrors the Notion page "BTR Update keywords to exclude", which is canonical.
EXCLUDED_KEYWORDS = [
    "international convention",
    "united nations framework convention on climate change",
    "ndc",
    "nationally determined contributions",
    "nationally determined contribution",
    "national adaptation plan",
    "national adaptation plans",
    "national communications",
    "national communication",
    "biennial reports",
    "biennial report",
    "biennial transparency reports",
    "biennial transparency report",
    "btr",
    "biennial update reports",
    "biennial update report",
    "long-term low greenhouse gas development strategy",
    "long-term low greenhouse gas development strategies",
    "long term low greenhouse gas development strategy",
    "long term low greenhouse gas development strategies",
    "LT-LEDS",
    "national greenhouse gas inventory",
    "adaptation communications",
    "adaptation communication",
    "national inventory reports",
    "national inventory report",
    "annex",
    "appendix",
    "nir",
    "unfccc",
    "fao",
    "iea",
    "ipcc",
    "unep",
    "undp",
    "unccd",
    "undrr",
    "united nations",
    "world bank",
    "energy balance",
    "paris agreement",
    "et al",
    "table",
    "chapter",
    "national inventory document",
    "national inventory documents",
    "nid",
    "transparency report",
    "assessment report",
    "inventory of greenhouse",
    "inventory of ghg",
    "national greenhouse gas inventories",
    "methodology",
    "inventory",
]
# Bounded by non-letters: substring matching hits "table" in "sustainable", and \b
# misses references glued to digits, like "NDC3" and "1BTR".
KEYWORD_PATTERN = re.compile(
    r"(?<![a-z])(?:"
    + "|".join(re.escape(k) for k in EXCLUDED_KEYWORDS)
    + r")s?(?![a-z])",  # allow a plural "s", e.g. "NDCs"
    re.IGNORECASE,
)
MATCHING_RULE = (
    "labelled_text is flagged keyword_exclude when it contains any keyword, "
    "case-insensitive, that is not directly preceded or followed by a letter (regex "
    "(?<![a-z])<keyword>s?(?![a-z])), allowing a plural 's'. So 'table' doesn't match 'Tabletop' or "
    "'sustainable', but 'ndc' does match 'NDC3' and 'btr' matches '1BTR'. "
    "nonalphabetic_exclude is set when labelled_text contains no alphabetic characters."
)

MENTION_COLUMNS = [
    "geography",
    "labelled_text",
    "relevant",
    "duplicate CCLW",
    "notes",
    "text",
    "page_number",
    "document_id",
    "document_name",
    "document_type",
    "document_source_url",
    "keyword_exclude",
    "nonalphabetic_exclude",
    "search_link",
    "search_link_with_geography",
]


# ------------------------------------------------------------------------ helpers


def read_ids_file(path: Path) -> pd.DataFrame:
    """One row per document id, with its trailing comment as ``label``."""
    rows = []
    for line in path.read_text().splitlines():
        doc_id, _, comment = line.partition("#")
        doc_id, comment = doc_id.strip(), comment.strip()
        if doc_id:
            rows.append(
                {
                    "document_id": doc_id,
                    "label": comment,
                    "is_annex": "(annex)" in comment.lower(),
                }
            )
    return pd.DataFrame(rows)


def load_geographies(cache_path: Path) -> dict[str, dict]:
    """ISO3 code -> {name, slug} from the public CPR API, cached in the run dir."""
    if not cache_path.exists():
        response = httpx.get(GEOGRAPHIES_URL, timeout=60)
        response.raise_for_status()
        cache_path.write_text(response.text)
    data = json.loads(cache_path.read_text())["data"]
    return {
        g["id"]: {"name": g["name"], "slug": g["slug"]}
        for g in data
        if g.get("type") == "country"
    }


def document_type(name: str, label: str = "") -> str:
    """NDC or BTR, from the document name (falling back to the ids-file label)."""
    for source in (name or "", label or ""):
        lowered = source.lower()
        if "transparency report" in lowered or re.search(r"\bbtr\b", lowered):
            return "BTR"
        if "nationally determined contribution" in lowered or re.search(
            r"\bndc\b", lowered
        ):
            return "NDC"
    return ""


def first_page(pages) -> Optional[int]:
    """1-indexed first page from 0-indexed PAGE_NUMBERS, or None."""
    pages = list(pages) if pages is not None else []
    return int(pages[0]) + 1 if pages else None


def search_link(text: str, geography_slug: Optional[str] = None) -> str:
    link = f"{SEARCH_URL}?q={quote_plus(text)}"
    if geography_slug:
        link += f"&l={quote_plus(geography_slug)}"
    return link


def add_document_fields(
    df: pd.DataFrame, geographies: dict[str, dict], ids: pd.DataFrame
) -> pd.DataFrame:
    """Add geography, document_type and page_number columns."""
    df = df.copy()
    labels = dict(zip(ids["document_id"], ids["label"]))

    def geo(codes, key):
        codes = list(codes) if codes is not None else []
        values = [geographies.get(c, {}).get(key, c) for c in codes]
        return "; ".join(v for v in values if v)

    df["geography"] = df["geographies"].map(lambda c: geo(c, "name"))
    df["geography_slug"] = df["geographies"].map(
        lambda c: geo(c[:1] if c is not None and len(c) else [], "slug")
    )
    df["document_type"] = [
        document_type(n, labels.get(d, ""))
        for n, d in zip(df["document_name"], df["document_id"])
    ]
    df["page_number"] = df["page_numbers"].map(first_page).astype("Int64")
    df["text"] = df["text"].map(lambda t: html.unescape(str(t)))
    return df


# ------------------------------------------------------------------------ build


def main(
    run_dir: Path = typer.Option(..., exists=True, help="Run directory."),
    bert: Path = typer.Option(..., exists=True, help="Combined output of predict.py."),
    ids_file: Path = typer.Option(..., exists=True, help="Requested document ids."),
    threshold: float = typer.Option(0.5, help="BERT threshold used in `filter`."),
    run_date: str = typer.Option(
        date.today().isoformat(), help="Date used in output names (YYYY-MM-DD)."
    ),
):
    """Write the mentions workbook to the run dir."""
    ids = read_ids_file(ids_file)
    geographies = load_geographies(run_dir / "geographies.json")

    bert_df = pd.read_parquet(bert) if bert.suffix == ".parquet" else pd.read_csv(bert)
    candidates = pd.read_parquet(run_dir / "candidates.parquet")
    duplicates = pd.read_parquet(run_dir / "duplicates.parquet")
    spans = pd.read_parquet(run_dir / "llm_spans.parquet")
    problems = _read_optional(run_dir / "llm_problems.parquet")
    model = _model_name(run_dir)

    # ---- exclusion flags
    spans = add_document_fields(spans, geographies, ids)
    spans["labelled_text"] = spans["labelled_text"].map(
        lambda t: html.unescape(str(t)).strip()
    )
    spans["keyword_exclude"] = spans["labelled_text"].map(
        lambda t: bool(KEYWORD_PATTERN.search(t))
    )
    spans["nonalphabetic_exclude"] = spans["labelled_text"].map(
        lambda t: not any(c.isalpha() for c in t)
    )
    spans["search_link"] = spans["labelled_text"].map(search_link)
    spans["search_link_with_geography"] = [
        search_link(t, s)
        for t, s in zip(spans["labelled_text"], spans["geography_slug"])
    ]
    for col in ("relevant", "duplicate CCLW", "notes"):
        spans[col] = ""
    spans = spans.sort_values(
        by=["geography", "document_id", "passage_idx", "start_index"]
    )
    excluded_mask = spans["keyword_exclude"] | spans["nonalphabetic_exclude"]
    mentions = spans.loc[~excluded_mask, MENTION_COLUMNS]
    excluded = spans.loc[excluded_mask, MENTION_COLUMNS]

    # ---- duplicates tab: each removed passage, with what its representative yielded
    duplicates = add_document_fields(duplicates, geographies, ids)
    rep_mentions: dict[str, str] = {
        str(pid): "; ".join(dict.fromkeys(group["labelled_text"]))
        for pid, group in spans.groupby("passage_id")
    }
    duplicates["representative_labelled_text"] = duplicates[
        "representative_passage_id"
    ].map(lambda pid: rep_mentions.get(pid, ""))
    duplicates = duplicates.sort_values(by=["geography", "document_id", "passage_idx"])
    duplicate_rows = duplicates[
        [
            "geography",
            "text",
            "page_number",
            "document_id",
            "document_name",
            "document_type",
            "document_source_url",
            "representative_passage_id",
            "representative_labelled_text",
            "passage_id",
        ]
    ]

    # ---- per-document summary (README tab)
    summary = _document_summary(
        ids, bert_df, candidates, spans, mentions, threshold, geographies
    )

    criteria = pd.DataFrame(
        [
            ("run date", run_date),
            ("passage classifier", "ModernBERT-base policy-mentions classifier"),
            ("BERT threshold", f"class_1_prob >= {threshold}"),
            (
                "threshold rationale",
                "BERT is a cost filter ahead of the LLM. Against LLM labels in March "
                "2026, 0.75 gave recall 0.63 and 0.5 gave recall 1.0; precision was "
                "0.16-0.23 at every threshold.",
            ),
            (
                "passages",
                "English passages longer than 20 characters from "
                "PRODUCTION.PUBLISHED.PASSAGES",
            ),
            (
                "de-duplication",
                "exact match on normalised text (HTML-unescaped, lower-cased, "
                "whitespace collapsed) within each document; the first occurrence is "
                "kept and the rest are listed in the 'duplicate rows' tab",
            ),
            ("span extraction model", model),
            (
                "span extraction concept",
                "'document mention': the title of a document, formal citation or "
                "loose reference; not organisations, acronyms, articles, frameworks, "
                "conventions, the Paris Agreement, IPCC Guidelines or a bare 'NDC'. "
                "Prompt instructs the model to be very conservative.",
            ),
            ("exclusion matching rule", MATCHING_RULE),
            ("excluded keywords", "; ".join(EXCLUDED_KEYWORDS)),
            ("page_number", "first page of the passage, 1-indexed"),
        ],
        columns=pd.Index(["criterion", "value"]),
    )

    readme = _readme(summary, ids, problems)

    stem = f"BTR_NDC_mentions_{run_date}"
    xlsx_path = run_dir / f"{stem}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        mentions.to_excel(writer, sheet_name="mentions", index=False)
        excluded.to_excel(writer, sheet_name="excluded mentions", index=False)
        duplicate_rows.to_excel(writer, sheet_name="duplicate rows", index=False)
        criteria.to_excel(writer, sheet_name="criteria", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
        summary.to_excel(
            writer, sheet_name="README", index=False, startrow=len(readme) + 2
        )
        _format_workbook(writer)

    console.print(f"✅ {len(mentions)} mentions, {len(excluded)} excluded")
    if n_realigned := int((~spans["llm_exact_match"]).sum()):
        console.print(
            f"[yellow]⚠️ {n_realigned} spans re-aligned because the LLM altered the "
            "text it tagged; check these for fabrication (llm_exact_match is False "
            "in llm_spans.parquet)[/]"
        )
    console.print(f"💾 {xlsx_path}")


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _model_name(run_dir: Path) -> str:
    cache = run_dir / "llm_cache.jsonl"
    if cache.exists():
        with cache.open() as f:
            for line in f:
                if line.strip():
                    return json.loads(line)["model"]
    return "unknown"


def _document_summary(
    ids: pd.DataFrame,
    bert_df: pd.DataFrame,
    candidates: pd.DataFrame,
    spans: pd.DataFrame,
    mentions: pd.DataFrame,
    threshold: float,
    geographies: dict[str, dict],
) -> pd.DataFrame:
    counts = pd.DataFrame(
        {
            "passages": bert_df.groupby("document_id").size(),
            "above_threshold": bert_df[bert_df["class_1_prob"] >= threshold]
            .groupby("document_id")
            .size(),
            "llm_passages_after_dedup": candidates.groupby("document_id").size(),
            "spans": spans.groupby("document_id").size(),
            "mentions": mentions.groupby("document_id").size(),
        }
    )
    summary = ids.merge(counts, left_on="document_id", right_index=True, how="left")
    for col in counts.columns:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["excluded"] = summary["spans"] - summary["mentions"]
    first_rows = bert_df.drop_duplicates("document_id")
    doc_names = dict(zip(first_rows["document_id"], first_rows["document_name"]))
    doc_geos = {
        d: "; ".join(
            geographies.get(x, {}).get("name", x) for x in (c if c is not None else [])
        )
        for d, c in zip(first_rows["document_id"], first_rows["geographies"])
    }
    summary["document_name"] = summary["document_id"].map(doc_names)
    summary["geography"] = summary["document_id"].map(doc_geos)

    def note(row) -> str:
        if row["passages"] == 0:
            return "zero passages returned from Snowflake"
        if row["mentions"] == 0:
            return "zero mentions"
        return ""

    summary["note"] = summary.apply(note, axis=1)
    return pd.DataFrame(
        summary[
            [
                "document_id",
                "label",
                "geography",
                "document_name",
                "is_annex",
                "passages",
                "above_threshold",
                "llm_passages_after_dedup",
                "spans",
                "excluded",
                "mentions",
                "note",
            ]
        ]
    )


def _readme(
    summary: pd.DataFrame, ids: pd.DataFrame, problems: pd.DataFrame
) -> pd.DataFrame:
    lines = [
        "One row per document mention found by the LLM. Read labelled_text first; "
        "text gives the surrounding passage for context.",
        "Fill in relevant, duplicate CCLW and notes. Rows in 'excluded mentions' were "
        "flagged by the keyword rules in 'criteria'; move any false exclusions back.",
        f"{len(ids)} documents requested; {int(ids['is_annex'].sum())} are annexes "
        "(marked is_annex below).",
        "Annexes included: Belize BTR annex (i00007650), Costa Rica BTR REDD annex "
        "(i00007895, replacing the superseded i00004765), Philippines NDC annex "
        "(i00007958).",
        "Not included: 18 unfccc.int landing-page stubs with 0-3 passages each, the "
        "Congo cover letter (i00006977) and the superseded Sao Tome and Principe "
        "NDC3.0 original (i00004888, replaced by the 2026 update i00007936).",
    ]
    zero_passages = summary.loc[summary["passages"] == 0, "label"].tolist()
    zero_mentions = summary.loc[
        (summary["passages"] > 0) & (summary["mentions"] == 0), "label"
    ].tolist()
    lines.append(
        "Documents with zero passages: " + ("; ".join(zero_passages) or "none")
    )
    lines.append(
        "Documents with zero mentions: " + ("; ".join(zero_mentions) or "none")
    )
    if len(problems):
        lines.append(
            f"{len(problems)} passages could not be processed by the LLM and are "
            "not represented."
        )
    return pd.DataFrame({"README": lines})


def _format_workbook(writer: pd.ExcelWriter) -> None:
    """Freeze header rows, add filters and set readable column widths."""
    widths = {"labelled_text": 45, "text": 90, "document_name": 40, "notes": 25}
    for name, sheet in writer.sheets.items():
        if name in ("README",):
            sheet.column_dimensions["A"].width = 60
            continue
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            width = widths.get(str(cell.value), 18)
            sheet.column_dimensions[cell.column_letter].width = width


if __name__ == "__main__":
    typer.run(main)
