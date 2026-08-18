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

# Per-group caps for the text-statistics pull. The three exclusive classes and
# the all-three class are taken in full, so figure 1 describes the whole corpus
# rather than a sample. `background` stays capped: it is millions of passages and
# only supplies the Dirichlet prior and the lexicon baseline, both of which are
# already estimated to well under a tenth of a point at this size.
SAMPLE_LIMITS = {
    "only_Q32": None,
    "only_Q911": None,
    "only_Q912": None,
    "all_three": None,
    "background": 60_000,
}

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


# Concept families for the co-occurrence heatmap, taken from the concept store
# hierarchy rather than from name guesses. Only members that carry a `primary`
# classifier reach `topics` (gotcha 8), so these are the subsets that actually
# appear in the data: 6 of Q47's descendants and 9 of Q672's.
#
# Q672 ("impacted group") itself has no classifier, so only its children fire.
# Note both Q911 and Q912 are formally subconcept_of Q32, and Q47 lists Q32 as a
# related concept, so any co-occurrence here is partly ontology, not discourse.
JUST_TRANSITION = {
    "Q47": "just transition",
    "Q58": "social inclusion",
    "Q53": "social protection",
    "Q68": "decent work",
    "Q1754": "aligning skills",
    "Q69": "green jobs",
    "Q1744": "legal safeguards for vulnerable groups",
}
IMPACTED_GROUPS = {
    "Q704": "women and minority genders",
    "Q695": "youth",
    "Q676": "marginalized ethnicity",
    "Q684": "indigenous people",
    "Q1167": "people with limited assets",
    "Q690": "people with health conditions",
    "Q701": "people on the move",
    "Q708": "elderly people",
    "Q1016": "sexual minority",
}
RELATED_CONCEPTS = {**JUST_TRANSITION, **IMPACTED_GROUPS}


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
        cap = SAMPLE_LIMITS[name]
        # hash(id) ordering makes a capped draw quasi-random and reproducible.
        order_clause = f"order by hash(p.id) limit {cap}" if cap else ""
        sql = f"""
            select '{name}' as grp, p.id, p.document_id, p.content, d.category
            from PRODUCTION.PUBLISHED.PASSAGES p
            join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
            where {PASSAGE_FILTER}
              and not d.is_principal and not d.is_collection
              and d.document_status = 'published'
              and ({predicate})
            {order_clause}
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


def pull_concept_cooccurrence(conn) -> pd.DataFrame:
    """
    Co-occurrence of each justice classifier with the related concept families.

    Returns per-concept co-occurrence counts plus the three justice marginals and
    the corpus total, so lift can be computed without a second query.
    """
    ids = ", ".join(f"'{c}'" for c in RELATED_CONCEPTS)
    sql = f"""
        with base as (
            select p.concept_ids,
                   {has("Q32")} as q32, {has("Q911")} as q911, {has("Q912")} as q912
            from PRODUCTION.PUBLISHED.PASSAGES p
            join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
            where {PASSAGE_FILTER}
              and not d.is_principal and not d.is_collection
              and d.document_status = 'published'
              and d.category != 'Litigation'
        ),
        tot as (
            select count(*) as n_all,
                   sum(iff(q32, 1, 0)) as n_q32,
                   sum(iff(q911, 1, 0)) as n_q911,
                   sum(iff(q912, 1, 0)) as n_q912
            from base
        )
        select c.value::string as concept_id,
               count(*) as n_concept,
               sum(iff(b.q32, 1, 0)) as with_q32,
               sum(iff(b.q911, 1, 0)) as with_q911,
               sum(iff(b.q912, 1, 0)) as with_q912,
               any_value(t.n_all) as n_all,
               any_value(t.n_q32) as n_q32,
               any_value(t.n_q911) as n_q911,
               any_value(t.n_q912) as n_q912
        from base b, lateral flatten(input => b.concept_ids) c, tot t
        where c.value::string in ({ids})
        group by 1
    """
    return query(conn, sql)


def pull_justice_timeline(conn) -> pd.DataFrame:
    """
    Justice-labelled passages per publication year and corpus group.

    Rows are disjoint on the exclusivity classes, so only_* plus multiple sums
    to the justice total. The dist/proc columns are a second, independent
    cross-tab that ignores Q32 entirely, for the distributive-versus-procedural
    view.

    National greenhouse gas inventory reports and common reporting tables are
    split out of UN submissions into their own group rather than dropped. They
    are emissions accounting rather than policy discourse: they enter the corpus
    at scale from 2023 (12k passages in 2022, 87k in 2023) and carry justice
    language on about 1% of passages, so pooled into UN submissions they mask a
    rise everywhere else. The title match is confined to UN submissions, since a
    handful of laws and policies *about* inventories are real policy text.

    Corporate disclosure is excluded: that dataset has not been refreshed this
    year, so its passages all sit in the past and would bend any trend.

    Malformed years (a few documents carry values like 22 and 223) and null
    dates are dropped; 2026 is a partial year.
    """
    inventory = (
        "(d.title ilike '%inventory%' or d.title ilike '%CRT%' "
        "or d.title ilike '%common reporting table%')"
    )
    sql = f"""
        select year(d.published_date) as yr,
               case when d.category = 'UN submission' and {inventory}
                         then 'Technical reporting'
                    when d.category in ('Law', 'Policy') then 'Law + Policy'
                    when d.category = 'UN submission' then 'UN submission'
                    else 'MCF project' end as corpus_group,
               count(*) as passages_total,
               sum(iff({has("Q32")} and not {has("Q911")} and not {has("Q912")}, 1, 0)) as only_q32,
               sum(iff(not {has("Q32")} and {has("Q911")} and not {has("Q912")}, 1, 0)) as only_q911,
               sum(iff(not {has("Q32")} and not {has("Q911")} and {has("Q912")}, 1, 0)) as only_q912,
               sum(iff(iff({has("Q32")}, 1, 0) + iff({has("Q911")}, 1, 0)
                       + iff({has("Q912")}, 1, 0) > 1, 1, 0)) as multiple,
               sum(iff({has("Q911")} and not {has("Q912")}, 1, 0)) as dist_only,
               sum(iff({has("Q912")} and not {has("Q911")}, 1, 0)) as proc_only,
               sum(iff({has("Q911")} and {has("Q912")}, 1, 0)) as dist_and_proc
        from PRODUCTION.PUBLISHED.PASSAGES p
        join PRODUCTION.PUBLISHED.DOCUMENTS d on d.id = p.document_id
        where {PASSAGE_FILTER}
          and not d.is_principal and not d.is_collection
          and d.document_status = 'published'
          and d.category in ('Law', 'Policy', 'UN submission',
                             'Multilateral Climate Fund project')
          and d.published_date is not null
          and year(d.published_date) between 1998 and 2026
        group by 1, 2
        order by 1, 2
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
        "concept_cooccurrence": pull_concept_cooccurrence,
        "justice_timeline": pull_justice_timeline,
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
