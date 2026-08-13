"""
Pull the raw data for the climate justice classifier analysis.

Three classifiers, all `primary` profile, all excluded from the Sabin
(litigation) corpus:

    Q32   climate justice        (overarching / generic justice language)
    Q911  distributive justice
    Q912  procedural justice

Everything lands in ./data as parquet so the analysis scripts can be re-run
without hitting Snowflake again.
"""

import sys
from pathlib import Path

import pandas as pd
import snowflake.connector as sc

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

CONCEPTS = {
    "Q32": "climate justice",
    "Q911": "distributive justice",
    "Q912": "procedural justice",
}

# Named documents for the deep dive: three 2026 NBSAPs, and both countries'
# current NDC and current long-term strategy. Superseded submissions are not
# included — see git history for the earlier NDC-vs-predecessor comparison.
DEEP_DIVE_DOCS = {
    "UNFCCC.document.i00007868.n0000": "Uganda NBSAP (2026)",
    "UNFCCC.document.i00007484.n0000": "Armenia NBSAP (2026)",
    "UNFCCC.document.i00007864.n0000": "Portugal NBSAP (2026)",
    "UNFCCC.document.i00006565.n0000": "Türkiye NDC 3.0 (Nov 2025)",
    "UNFCCC.document.i00000391.n0000": "Türkiye LT-LEDS (Nov 2024)",
    "UNFCCC.document.i00004795.n0000": "Australia 2035 NDC (Sep 2025)",
    "UNFCCC.document.i00007760.n0000": "Australia LT-LEDS3 (Nov 2025)",
}

# Passages sampled per class for the text-statistics work. Sampling is by
# hash(id) so the same rows come back on every run.
SAMPLE_PER_CLASS = 60_000

# Standard passage hygiene for the corpus-wide comparisons: body text only, no
# short fragments (gotchas 2 & 3). Corpora differ a lot in how much page
# furniture they carry, so holding content_type fixed keeps the rates comparable.
PASSAGE_FILTER = "p.content_type = 'Text' and length(p.content) >= 20"

# For single-document work we keep every content type. The classifiers ran on
# all passages regardless of type, so restricting to 'Text' would understate
# where justice language actually sits — a lot of it is in List items.
#
# NB: do not use DOCUMENTS.passage_count as a denominator anywhere. It is rolled
# up from the *v1* passages table, not PASSAGES: for all four deep-dive
# documents it matches PASSAGES_V1 row counts exactly and overstates the real
# v2 row count by 7-11x. Always count PASSAGES rows directly.
DOC_FILTER = "length(p.content) >= 20"


def has(cid: str, col: str = "p.concept_ids") -> str:
    return f"array_contains('{cid}'::variant, {col})"


ANY_JUSTICE = " or ".join(has(c) for c in CONCEPTS)
NO_JUSTICE = " and ".join(f"not {has(c)}" for c in CONCEPTS)


def query(conn, sql: str) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    # Snowflake hands 0/1 flags back as int8. A groupby sum over more than 127
    # rows then silently wraps negative, so widen them on the way in.
    narrow = df.select_dtypes(include=["int8", "int16"]).columns
    return df.astype({c: "int32" for c in narrow})


def pull_text_samples(conn) -> pd.DataFrame:
    """Sampled passage text per exclusivity class, for TF-IDF / log-odds."""
    # "Exclusive" classes isolate what each classifier sees that the other two
    # do not — the sharpest read on how they differ from each other.
    groups = {
        "only_Q32": f"{has('Q32')} and not {has('Q911')} and not {has('Q912')}",
        "only_Q911": f"not {has('Q32')} and {has('Q911')} and not {has('Q912')}",
        "only_Q912": f"not {has('Q32')} and not {has('Q911')} and {has('Q912')}",
        "all_three": f"{has('Q32')} and {has('Q911')} and {has('Q912')}",
        # Background = everything the justice classifiers did not fire on. Note
        # this includes passages no classifier ran on at all (gotcha 8), so it is
        # a corpus baseline, not a curated negative set.
        "background": f"(p.concept_ids is null or ({NO_JUSTICE}))",
    }
    frames = []
    for name, predicate in groups.items():
        sql = f"""
            select '{name}' as grp, p.id, p.document_id, p.content, d.category
            from PRODUCTION.PUBLISHED.PASSAGES p
            join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
            where {PASSAGE_FILTER}
              and not d.is_principal and not d.is_collection
              and d.document_status = 'published'
              and ({predicate})
            order by hash(p.id)
            limit {SAMPLE_PER_CLASS}
        """
        df = query(conn, sql)
        print(f"  {name:12s} {len(df):>7,} passages")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pull_corpus_rates(conn) -> pd.DataFrame:
    """Justice hit-rate per corpus (category), with denominators."""
    sql = f"""
        select d.category,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912,
               sum(iff({ANY_JUSTICE}, 1, 0)) as any_justice,
               count(distinct d.id) as docs_total,
               count(distinct iff({ANY_JUSTICE}, d.id, null)) as docs_any_justice
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
        group by 1
        order by passages_total desc
    """
    return query(conn, sql)


def pull_region_rates(conn) -> pd.DataFrame:
    """
    Justice hit-rate per World Bank region, excluding litigation.

    Uses `regions` rather than `geographies` so the grain stays consistent
    (gotcha 11). Documents tagged with several regions fan out, so this is a
    per-(document, region) rate, not a partition of the corpus.

    Litigation is dropped because all three classifiers carry
    `dont_run_on: ["sabin"]`, and every Litigation document is a Sabin one
    (17,347 of 17,347). Leaving it in reads as "North America barely mentions
    justice" when the truth is the classifiers never ran there.
    """
    sql = f"""
        select r.value::string as region,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id,
             lateral flatten(input => d.regions) r
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category != 'Litigation'
        group by 1
        having count(*) >= 20000
        order by passages_total desc
    """
    return query(conn, sql)


def pull_region_rates_by_corpus(conn) -> pd.DataFrame:
    """
    Region rates held within a single corpus, as a confounding check.

    The pooled regional gap could be an artefact of which corpora cover which
    regions. Splitting by category tests that: the ranking survives inside both
    Policy and UN submission, though the size of the gap does not.
    """
    sql = f"""
        select d.category, r.value::string as region,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id,
             lateral flatten(input => d.regions) r
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category in ('Policy', 'UN submission')
        group by 1, 2
        having count(*) >= 15000
        order by 1, q911 / count(*) desc
    """
    return query(conn, sql)


def pull_region_rates_ex_mcf(conn) -> pd.DataFrame:
    """
    Region rates with Multilateral Climate Fund projects also removed.

    MCF projects are by far the justice-densest corpus and their geographic
    footprint is concentrated in exactly the regions that top the pooled chart,
    so they are the obvious alternative explanation for the regional gradient.
    """
    sql = f"""
        select r.value::string as region,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id,
             lateral flatten(input => d.regions) r
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category not in ('Litigation', 'Multilateral Climate Fund project')
        group by 1
        having count(*) >= 20000
        order by passages_total desc
    """
    return query(conn, sql)


def pull_country_justice_text(conn) -> pd.DataFrame:
    """
    Justice-labelled passage text from Australian and Turkish documents.

    Feeds the country vocabulary comparison. Litigation excluded (classifiers
    never ran on it); category kept so the law-only subset can be split out.
    """
    sql = f"""
        select case when array_contains('country::AUS'::variant, d.geographies)
                    then 'Australia' else 'Türkiye' end as country,
               d.category, d.title, p.content,
               iff({has("Q32")}, 1, 0) as q32,
               iff({has("Q911")}, 1, 0) as q911,
               iff({has("Q912")}, 1, 0) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category != 'Litigation'
          and ({ANY_JUSTICE})
          and (array_contains('country::AUS'::variant, d.geographies)
               or array_contains('country::TUR'::variant, d.geographies))
          -- documents tagged to both countries would double-count
          and not (array_contains('country::AUS'::variant, d.geographies)
                   and array_contains('country::TUR'::variant, d.geographies))
    """
    return query(conn, sql)


def pull_country_rates(conn) -> pd.DataFrame:
    """Per-country justice mix, excluding litigation. Countries only (gotcha 11)."""
    sql = f"""
        select c.value::string as country,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912,
               sum(iff({ANY_JUSTICE}, 1, 0)) as any_justice
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id,
             lateral flatten(input => d.countries) c
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category != 'Litigation'
        group by 1
        having count(*) >= 15000
        order by passages_total desc
    """
    return query(conn, sql)


def pull_overlap(conn) -> pd.DataFrame:
    """Seven-cell breakdown of which classifiers co-fire on the same passage."""
    sql = f"""
        select sum(iff(a and not b and not c, 1, 0)) as only_q32,
               sum(iff(not a and b and not c, 1, 0)) as only_q911,
               sum(iff(not a and not b and c, 1, 0)) as only_q912,
               sum(iff(a and b and not c, 1, 0)) as q32_q911,
               sum(iff(a and not b and c, 1, 0)) as q32_q912,
               sum(iff(not a and b and c, 1, 0)) as q911_q912,
               sum(iff(a and b and c, 1, 0)) as all_three,
               count(*) as any_justice,
               sum(count(*)) over () as _t
        from (
            select {has("Q32")} as a, {has("Q911")} as b, {has("Q912")} as c
            from PRODUCTION.PUBLISHED.PASSAGES p
            join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
            where {PASSAGE_FILTER}
              and not d.is_principal and not d.is_collection
              and d.document_status = 'published'
        )
        where a or b or c
    """
    return query(conn, sql)


def pull_deep_dive_passages(conn) -> pd.DataFrame:
    """
    Every body passage of the four named documents, with per-class flags.

    `idx` is kept so we can plot where in each document the justice language
    sits; it indexes all blocks, not just Text ones, so gaps are expected.
    """
    ids = ", ".join(f"'{i}'" for i in DEEP_DIVE_DOCS)
    sql = f"""
        select p.document_id, d.title, p.idx, p.content, p.page_numbers,
               iff({has("Q32")}, 1, 0) as q32,
               iff({has("Q911")}, 1, 0) as q911,
               iff({has("Q912")}, 1, 0) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where p.document_id in ({ids})
          and {DOC_FILTER}
        order by p.document_id, p.idx
    """
    return query(conn, sql)


def pull_law_rates(conn) -> pd.DataFrame:
    """Per-law justice rates for every published Australian and Turkish law."""
    sql = f"""
        select case when array_contains('country::AUS'::variant, d.geographies)
                    then 'Australia' else 'Türkiye' end as country,
               d.id, d.title, d.published_date,
               count(*) as passages_total,
               sum(iff({has("Q32")}, 1, 0)) as q32,
               sum(iff({has("Q911")}, 1, 0)) as q911,
               sum(iff({has("Q912")}, 1, 0)) as q912,
               sum(iff({ANY_JUSTICE}, 1, 0)) as any_justice
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where {DOC_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category = 'Law'
          and (array_contains('country::AUS'::variant, d.geographies)
               or array_contains('country::TUR'::variant, d.geographies))
        group by 1, 2, 3, 4
        order by country, passages_total desc
    """
    return query(conn, sql)


def pull_law_passages(conn) -> pd.DataFrame:
    """Justice-labelled passages from AUS/TUR laws, for qualitative reading."""
    sql = f"""
        select case when array_contains('country::AUS'::variant, d.geographies)
                    then 'Australia' else 'Türkiye' end as country,
               d.title, p.content,
               iff({has("Q32")}, 1, 0) as q32,
               iff({has("Q911")}, 1, 0) as q911,
               iff({has("Q912")}, 1, 0) as q912
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where {DOC_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category = 'Law'
          and (array_contains('country::AUS'::variant, d.geographies)
               or array_contains('country::TUR'::variant, d.geographies))
          and ({ANY_JUSTICE})
    """
    return query(conn, sql)


def main() -> None:
    conn = sc.connect(connection_name="cpr")
    pulls = {
        "corpus_rates": pull_corpus_rates,
        "overlap": pull_overlap,
        "region_rates": pull_region_rates,
        "region_rates_by_corpus": pull_region_rates_by_corpus,
        "region_rates_ex_mcf": pull_region_rates_ex_mcf,
        "country_justice_text": pull_country_justice_text,
        "country_rates": pull_country_rates,
        "law_rates": pull_law_rates,
        "law_passages": pull_law_passages,
        "deep_dive_passages": pull_deep_dive_passages,
        "text_samples": pull_text_samples,
    }
    only = set(sys.argv[1:])
    for name, fn in pulls.items():
        if only and name not in only:
            continue
        print(f"\n{name}...")
        df = fn(conn)
        df.to_parquet(DATA / f"{name}.parquet", index=False)
        print(f"  -> {name}.parquet  ({len(df):,} rows)")
    conn.close()


if __name__ == "__main__":
    main()
