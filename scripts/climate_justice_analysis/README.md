# Climate justice classifier analysis

Written by Claude (Anthropic) with Anne Sietsma, August 2026.

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

Charts into `figures/`. This script carries a PEP 723 inline dependency header
and resolves its own isolated environment, so it adds nothing to the shared repo
dependencies.

PNG files are committed. SVG copies are written alongside them but gitignored:
each is ~400KB of machine-generated XML and together they were 95% of the diff.
Regenerate them with the same command when you need vectors for design work.

Figures are sized for slides — landscape or square, body text at 11pt, nothing
below about 8pt.

## House style

Figures follow the CPR presentation guidelines: white background, Inter Tight
for titles and Inter for everything else, the brand palette, and the wordmark
bottom-left. `house_style.py` holds all of it.

Two deviations, both for legibility, both noted in that file:

- The brand's designated data-viz colours are Cardboard, Mustard, Forest and
  Green. Inky Blue against Forest is only dE 21.5 under the worst colour-vision
  deficiency because both are dark, so the categorical trio is Inky Blue /
  Mustard / Green at dE 28.6.
- Mustard and Green are fine as fills but fail as text on white (2.1:1 and
  2.6:1), so text set in a series colour uses a darkened cousin.

The brand fonts ship as variable TTFs and matplotlib only loads a variable
font's default instance, so `house_style.apply()` cuts static Regular/SemiBold/
Bold instances with fontTools on first run and caches them in `assets/fonts/`.
If the fonts are not installed it falls back to system sans and says so.

Captions follow a fixed convention: the subtitle says how to read the marks,
the footnote carries only caveats that change how the numbers should be taken,
then the source line. The argument belongs in FINDINGS.md.

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
- Corporate disclosure is excluded from the time series (fig 9) but kept in the
  corpus heatmap (fig 3): that dataset has not been updated this year, so its
  passages all sit in the past and would distort a trend.
- Figure 9's headline rests on a standardisation, not a raw trend. Hold the 2016
  corpus mix fixed and 2025 reads 37.2% against 2016's 33.1%. Do not publish the
  raw share series without that correction.
- **Never use `DOCUMENTS.passage_count` as a denominator here** — it is rolled
  up from the v1 passages table and overstates the real count 7–11×. Count
  `PASSAGES` rows directly. See FINDINGS.md § Data notes.

## Layout

```
house_style.py    CPR palette, typography, logo, caption conventions
pull_data.py      Snowflake -> data/*.parquet
text_stats.py     log-odds + TF-IDF distinctiveness -> results/
specificity.py    lexicon yardstick for each classifier -> results/
viz.py            all twelve figures -> figures/  (PEP 723, isolated env)
FINDINGS.md       write-up
```
