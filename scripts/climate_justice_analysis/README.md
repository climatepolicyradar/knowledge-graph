# Climate justice classifier analysis

Exploratory analysis of the three climate justice classifiers — `Q32` (climate
justice), `Q911` (distributive justice), `Q912` (procedural justice) — for the
Topics blog post. Results in [FINDINGS.md](FINDINGS.md), charts in `figures/`.

## Running it

```bash
uv run python scripts/climate_justice_analysis/pull_data.py
```

Pulls everything from Snowflake into `data/*.parquet` (connection `cpr`, SSO).
Pass pull names as arguments to refresh only some of them, e.g.
`pull_data.py corpus_rates law_rates`.

```bash
uv run python scripts/climate_justice_analysis/text_stats.py
uv run python scripts/climate_justice_analysis/specificity.py
```

Word statistics into `results/*.csv`. These use the shared repo environment
(pandas + scikit-learn, both already dependencies).

```bash
cd scripts/climate_justice_analysis && uv run --script viz.py
```

Charts into `figures/` as PNG (200 dpi) and SVG. This one carries a PEP 723
inline dependency header and resolves its own isolated environment, so it adds
nothing to the shared repo dependencies.

## Notes on method

- Corpus-wide comparisons filter to `content_type = 'Text'` and ≥20 characters,
  because corpora differ a lot in how much page furniture they carry. The
  single-document and law analyses keep every passage type, since the
  classifiers ran on all of them and much of the justice language in ANDCs and
  NBSAPs sits in list items.
- Text sampling is `order by hash(p.id)`, so re-running returns the same rows.
- One- and two-word terms are ranked in separate vocabularies. Pooled, a bigram
  is always rarer than its parts and the Dirichlet prior shrinks it harder, so
  almost none survive into the top slots.
- `fetch_pandas_all` returns Snowflake 0/1 flags as **int8**, and a groupby
  `sum()` over more than 127 rows silently wraps negative. `pull_data.query()`
  widens them to int32 on the way in.
- **Litigation is excluded everywhere.** All three classifier specs carry
  `dont_run_on: ["sabin"]` and every published Litigation document is a Sabin
  record, so its 0% is an absence of inference, not an absence of justice
  language. Left in, it drags North America to near zero and manufactures a
  finding.
- Figure 4 additionally reports a variant with Multilateral Climate Fund
  projects removed, since that corpus is both the justice-densest and
  concentrated in the leading regions.
- **Never use `DOCUMENTS.passage_count` as a denominator here** — it is rolled
  up from the v1 passages table and overstates the real count 7–11×. Count
  `PASSAGES` rows directly. See FINDINGS.md § Data notes.

## Layout

```
pull_data.py      Snowflake -> data/*.parquet
text_stats.py     log-odds + TF-IDF distinctiveness -> results/
specificity.py    lexicon yardstick for each classifier -> results/
viz.py            all seven figures -> figures/   (PEP 723, isolated env)
FINDINGS.md       write-up
```
