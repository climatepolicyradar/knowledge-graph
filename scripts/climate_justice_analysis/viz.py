# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "pandas>=2.2", "numpy>=1.26", "pyarrow>=16",
#                 "svgpath2mpl>=1.0", "fonttools>=4.50"]
# ///
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
#
# The pandas and scipy stubs do not narrow well here: pyright reads a Series
# as an ndarray (losing .str/.head) and a sparse matrix as a scalar (losing
# .toarray()). Every such call is exercised on each run, so the suppression is
# scoped to these analysis scripts rather than applied repo-wide.
"""
Figures for the climate justice classifier analysis.

Run with `uv run --script viz.py` so it resolves its own dependencies and
leaves the shared repo environment alone.

House style — palette, typography, logo, caption conventions — lives in
house_style.py. Captions here are deliberately terse: the subtitle says how to
read the marks, the footnote carries only caveats that change how the numbers
should be taken. The argument belongs in FINDINGS.md.

Every figure still carries its exact values as text so it can be rebuilt in a
design tool without going back to the data.
"""

from pathlib import Path

import house_style as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from house_style import (
    GRID,
    INK,
    INK2,
    INK3,
    ink_on,
    sequential,
    text_colour,
    titled,
)
from matplotlib.patches import Rectangle

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

hs.apply()

Q32, Q911, Q912 = hs.Q32_C, hs.Q911_C, hs.Q912_C
COLORS = {"Q32": Q32, "Q911": Q911, "Q912": Q912}
LABELS = {
    "Q32": "Climate justice",
    "Q911": "Distributive justice",
    "Q912": "Procedural justice",
}
COUNTRY_COLORS = {"Australia": hs.COUNTRY_A, "Türkiye": hs.COUNTRY_B}


def save(fig, name: str) -> None:
    hs.save(fig, name, FIG)


# ---------------------------------------------------------------- figure 1
def fig_vocabulary() -> None:
    """
    Top distinguishing terms per classifier, by log-odds z.

    Unigrams and bigrams are ranked in separate vocabularies and shown as two
    blocks. Pooled, a bigram is always rarer than its parts and the prior shrinks
    it harder, so almost none survived into the top slots — "civil society" and
    "local communities" were invisible behind "community" and "local".
    """
    lo = pd.read_csv(RESULTS / "log_odds_by_class.csv")
    keys = ["only_Q32", "only_Q911", "only_Q912"]
    n_uni, n_bi = 10, 5
    # Derived from the data so the caption cannot go stale.
    raw = pd.read_parquet(DATA / "text_samples.parquet", columns=["GRP", "CONTENT"])
    raw["CONTENT"] = (
        raw.CONTENT.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    counts = raw.drop_duplicates(["GRP", "CONTENT"]).GRP.value_counts().to_dict()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 10.4))
    for ax, key in zip(axes, keys):
        cid = key.split("_")[1]
        uni = lo[(lo.cls == key) & (lo.ngram == 1)].head(n_uni)
        bi = lo[(lo.cls == key) & (lo.ngram == 2)].head(n_bi)
        # Bigrams sit in a block below the unigrams, with a gap between.
        sub = pd.concat([uni, bi], ignore_index=True)
        rows = list(range(n_uni)) + [r + 1.0 for r in range(n_uni, n_uni + len(bi))]
        y = np.array([max(rows) - r for r in rows])
        zmax = float(sub.z.max())

        ax.barh(y, sub.z, height=0.52, color=COLORS[cid], linewidth=0)
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.text(
                -zmax * 0.045,
                yi + 0.16,
                r.term,
                va="center",
                ha="right",
                fontsize=11.5,
                color=INK,
            )
            ax.text(
                -zmax * 0.045,
                yi - 0.22,
                f"{r.rate_per_10k_in_class:.0f} vs {r.rate_per_10k_in_others:.0f} per 10k",
                va="center",
                ha="right",
                fontsize=8.2,
                color=INK3,
            )
            ax.text(
                r.z + zmax * 0.02,
                yi,
                f"{r.z:.0f}",
                va="center",
                ha="left",
                fontsize=10.1,
                color=INK2,
                weight="bold",
            )
        gap_y = max(rows) - n_uni + 0.5
        ax.text(
            -zmax * 1.26,
            gap_y,
            "two-word terms, ranked separately",
            fontsize=8.6,
            color=INK3,
            va="center",
            ha="left",
            style="italic",
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(-1.1, max(y) + 0.6)
        ax.set_xlim(-zmax * 1.28, zmax * 1.12)
        ax.set_title(
            f"{LABELS[cid]}  ({cid})",
            color=text_colour(COLORS[cid]),
            weight="bold",
            fontsize=13.8,
            pad=16,
            loc="left",
        )
        ax.spines[:].set_visible(False)
        ax.text(
            -zmax * 1.26,
            -0.95,
            "bar length = log-odds z vs the other two classifiers",
            fontsize=8.6,
            color=INK3,
            va="center",
            ha="left",
        )
    fig.subplots_adjust(top=0.815, bottom=0.075, wspace=0.20)
    titled(
        fig,
        "Three justice classifiers, three vocabularies",
        "Bar length is how strongly a term marks out one classifier against the other two. Single words above the rule,\ntwo-word terms below, ranked separately. Grey figures are the term's rate per 10,000 words, this class vs the others.",
        f"Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), z-scored. Computed on every exclusive "
        f"passage of each class, not a sample: {counts['only_Q32']:,} / {counts['only_Q911']:,} / {counts['only_Q912']:,}.",
    )
    save(fig, "01_vocabulary_fingerprint")


# ---------------------------------------------------------------- figure 2
def fig_overlap() -> None:
    """
    UpSet-style view of which classifiers co-fire.

    A Venn was tried first and rejected: with three overlapping sets the cells
    cannot be drawn to scale, so circle area contradicts the printed numbers,
    and every cell needs its own denominator caption to be readable. An UpSet
    plot puts the seven disjoint combinations on one common bar scale, which is
    the actual comparison, and states each classifier's total separately.
    """
    o = pd.read_parquet(DATA / "overlap.parquet").iloc[0]
    total = int(o.ANY_JUSTICE)
    cids = ["Q32", "Q911", "Q912"]
    combos = [
        (("Q911",), int(o.ONLY_Q911)),
        (("Q32", "Q911"), int(o.Q32_Q911)),
        (("Q32", "Q911", "Q912"), int(o.ALL_THREE)),
        (("Q32",), int(o.ONLY_Q32)),
        (("Q912",), int(o.ONLY_Q912)),
        (("Q911", "Q912"), int(o.Q911_Q912)),
        (("Q32", "Q912"), int(o.Q32_Q912)),
    ]
    combos.sort(key=lambda c: -c[1])
    totals = {
        "Q32": int(o.ONLY_Q32 + o.Q32_Q911 + o.Q32_Q912 + o.ALL_THREE),
        "Q911": int(o.ONLY_Q911 + o.Q32_Q911 + o.Q911_Q912 + o.ALL_THREE),
        "Q912": int(o.ONLY_Q912 + o.Q32_Q912 + o.Q911_Q912 + o.ALL_THREE),
    }

    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.5, 1], width_ratios=[1.5, 3.3], hspace=0.06, wspace=0.04
    )
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)

    x = np.arange(len(combos))
    heights = [c[1] for c in combos]
    # A bar takes the colour of its classifier when only one fired, and stays
    # neutral when the passage carries several — the point of the grey bars is
    # that no single classifier owns them.
    bar_colors = [COLORS[c[0][0]] if len(c[0]) == 1 else hs.NEUTRAL for c in combos]
    ax_bar.bar(x, heights, width=0.62, color=bar_colors, linewidth=0)
    for xi, (sets, n) in zip(x, combos):
        ax_bar.text(
            xi,
            n + total * 0.030,
            f"{n:,}",
            ha="center",
            va="bottom",
            fontsize=13.8,
            weight="bold",
            color=INK,
        )
        ax_bar.text(
            xi,
            n + total * 0.010,
            f"{100 * n / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.1,
            color=INK2,
            weight="bold",
        )
    ax_bar.set_ylim(0, max(heights) * 1.24)
    ax_bar.set_ylabel("passages", fontsize=10.8, color=INK2)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(labelbottom=False, labelsize=8)
    ax_bar.yaxis.set_major_formatter(lambda v, _: f"{int(v):,}")

    # Dot matrix: filled = that classifier fired for this combination.
    for xi, (sets, _) in zip(x, combos):
        rows = [2 - cids.index(c) for c in sets]
        ax_mat.plot(
            [xi, xi],
            [min(rows), max(rows)],
            color=INK2,
            linewidth=1.6,
            zorder=1,
            solid_capstyle="round",
        )
        for r_i, cid in enumerate(cids):
            row = 2 - r_i
            on = cid in sets
            ax_mat.scatter(
                [xi],
                [row],
                s=170,
                zorder=2,
                color=COLORS[cid] if on else GRID,
                linewidth=0,
            )
    ax_mat.set_ylim(-0.8, 2.6)
    ax_mat.set_xlim(-0.6, len(combos) - 0.4)
    ax_mat.axis("off")

    # Set-size bars, mirrored left, sharing the matrix rows.
    tmax = max(totals.values())
    for r_i, cid in enumerate(cids):
        row = 2 - r_i
        ax_set.barh([row], [totals[cid]], height=0.30, color=COLORS[cid], linewidth=0)
        ax_set.text(
            tmax * 2.02,
            row + 0.30,
            LABELS[cid],
            ha="left",
            va="center",
            fontsize=12.0,
            weight="bold",
            color=text_colour(COLORS[cid]),
        )
        ax_set.text(
            tmax * 2.02,
            row + 0.02,
            f"{cid} · {totals[cid]:,} passages",
            ha="left",
            va="center",
            fontsize=9.1,
            color=INK3,
        )
        ax_set.text(
            tmax * 2.02,
            row - 0.24,
            f"{100 * totals[cid] / total:.0f}% of all justice passages",
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK3,
        )
    ax_set.set_xlim(tmax * 2.10, 0)
    ax_set.axis("off")

    fig.subplots_adjust(top=0.865, bottom=0.10, left=0.045, right=0.985)
    titled(
        fig,
        "Distributive justice does most of the work, and usually works alone",
        "Each bar is one of the seven ways the three classifiers can combine on a passage. Bars are disjoint and sum to 100%.\nThe smaller bars on the left are each classifier's own total, which do overlap.",
        "Coloured bars are passages where a single classifier fired; neutral bars carry more than one label.",
    )
    save(fig, "02_classifier_overlap")


# ---------------------------------------------------------------- figure 3
def fig_corpus_heatmap() -> None:
    """
    Corpus x classifier hit rate, shaded within each column.

    Litigation is excluded entirely: all three specs carry
    `dont_run_on: ["sabin"]` and every Litigation document is a Sabin record.

    One number per cell — the share of that corpus's own passages, so corpus
    size is divided out — and the shade tracks that same number. Colour is
    scaled *within each column*, because the three classifiers have very
    different base rates: on a shared scale distributive justice, three times
    the size of procedural justice, made every other column look empty. The
    cost is that shades compare down a column, not across one, which the
    caption says.
    """
    c = pd.read_parquet(DATA / "corpus_rates.parquet").set_index("CATEGORY")
    # Litigation only: the classifiers were never run on it, so its zero is an
    # absence of inference rather than an absence of justice language.
    c = c.drop(index="Litigation", errors="ignore")
    order = [
        "Multilateral Climate Fund project",
        "Policy",
        "Report",
        "UN submission",
        "Law",
        "Corporate Disclosure",
    ]
    c = c.loc[[o for o in order if o in c.index][::-1]]
    cids = ["Q32", "Q911", "Q912"]
    rates = pd.DataFrame({k: 100 * c[k] / c.PASSAGES_TOTAL for k in cids})
    baseline = {k: 100 * c[k].sum() / c.PASSAGES_TOTAL.sum() for k in cids}
    vmax = float(rates.to_numpy().max())

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    for i, corpus in enumerate(rates.index):
        for j, cid in enumerate(cids):
            v = rates.loc[corpus, cid]
            col = sequential(v, vmax)
            ax.add_patch(Rectangle((j, i), 0.96, 0.94, facecolor=col, linewidth=0))
            ax.text(
                j + 0.48,
                i + 0.47,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=19.2,
                weight="bold",
                color=ink_on(col),
            )
        ax.text(
            -0.16,
            i + 0.60,
            corpus,
            ha="right",
            va="center",
            fontsize=12.0,
            color=INK,
            weight="bold",
        )
        ax.text(
            -0.16,
            i + 0.30,
            f"{int(c.loc[corpus, 'PASSAGES_TOTAL']):,} passages · "
            f"{int(c.loc[corpus, 'DOCS_TOTAL']):,} docs",
            ha="right",
            va="center",
            fontsize=9.1,
            color=INK3,
        )
    for j, cid in enumerate(cids):
        ax.text(
            j + 0.48,
            len(rates) + 0.30,
            LABELS[cid],
            ha="center",
            va="bottom",
            fontsize=12.6,
            weight="bold",
            color=text_colour(COLORS[cid]),
        )
        ax.text(
            j + 0.48,
            len(rates) + 0.14,
            f"{cid} · {baseline[cid]:.1f}% overall",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=INK3,
        )

    ax.set_xlim(-2.55, 3.05)
    ax.set_ylim(-0.25, len(rates) + 0.80)
    ax.axis("off")
    fig.subplots_adjust(top=0.855, bottom=0.10)
    titled(
        fig,
        "Funding documents talk about justice; laws do not",
        "Each cell is the share of that corpus's own passages carrying the label, so corpus size is divided out.\nOne shared colour scale across the whole table, so any two cells are directly comparable.",
        "Litigation is excluded: the classifiers were never run on that corpus, so its zero would be an absence of inference rather than of justice language.",
    )
    save(fig, "03_corpus_heatmap")


# ---------------------------------------------------------------- figure 4
def fig_regions() -> None:
    """
    Regional profile, with and without the justice-densest corpus.

    Multilateral Climate Fund projects are the densest corpus by a wide margin
    and their geographic footprint is concentrated in exactly the regions that
    top the pooled chart, so they are the obvious alternative explanation for
    the gradient. The right panel removes them.
    """
    cids = ["Q32", "Q911", "Q912"]
    panels = []
    for fname, label in [
        ("region_rates", "All corpora"),
        ("region_rates_ex_mcf", "Excluding Multilateral Climate Fund projects"),
    ]:
        r = pd.read_parquet(DATA / f"{fname}.parquet").set_index("REGION")
        rates = pd.DataFrame({k: 100 * r[k] / r.PASSAGES_TOTAL for k in cids})
        panels.append((label, r, rates))

    order = panels[0][2].sort_values("Q911", ascending=True).index.tolist()
    xmax = max(p[2].to_numpy().max() for p in panels) * 1.30

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.8), sharey=True)
    h = 0.24
    y = np.arange(len(order))
    for ax, (label, r, rates) in zip(axes, panels):
        rates = rates.loc[order]
        for k, cid in enumerate(cids):
            offs = (1 - k) * h
            vals = rates[cid].to_numpy()
            ax.barh(
                y + offs,
                vals,
                height=h * 0.86,
                color=text_colour(COLORS[cid]),
                linewidth=0,
                label=LABELS[cid],
            )
            for yi, v in zip(y + offs, vals):
                ax.text(
                    v + xmax * 0.012,
                    yi,
                    f"{v:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=10.1,
                    weight="bold",
                    color=INK2,
                )
        ax.set_xlim(0, xmax)
        ax.set_title(label, loc="left", fontsize=12.6, weight="bold", color=INK, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel(
            "% of body-text passages carrying the label",
            fontsize=10.6,
            color=INK2,
            labelpad=8,
        )
        for yi, idx in zip(y, order):
            ax.text(
                xmax * 0.995,
                yi - 0.36,
                f"{int(r.loc[idx, 'PASSAGES_TOTAL']):,} passages",
                ha="right",
                va="center",
                fontsize=8.2,
                color=INK3,
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=12.2, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=11.5,
        ncols=3,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.845),
    )

    fig.subplots_adjust(top=0.845, bottom=0.13, left=0.16, wspace=0.07)
    titled(
        fig,
        "Removing the fund documents narrows the gap but does not close it",
        "Share of passages carrying each label, by World Bank region. The right panel repeats the left with multilateral\nclimate fund projects removed — the corpus most likely to be driving the gradient.",
        "Documents tagged to several regions count once in each, so rows do not partition the corpus. Litigation excluded.",
    )
    save(fig, "04_regions")


# ---------------------------------------------------------------- figure 5
def _deep_dive(order, name, title, how, figsize) -> None:
    """
    Where justice language sits inside six flagship documents.

    Tick width is 100/n of the document rather than a fixed line width, so each
    passage occupies the same *share* of its bar. With fixed-width ticks a
    24-passage NDC and a 402-passage NBSAP looked equally dense at very
    different real rates. A small floor keeps single passages visible in the
    longest document; consecutive hits merge into a solid block, which is the
    honest reading of a run of justice-labelled paragraphs.
    """
    d = pd.read_parquet(DATA / "deep_dive_passages.parquet")
    cids = ["Q32", "Q911", "Q912"]
    fig, axes = plt.subplots(len(order), 1, figsize=figsize)
    for ax, (doc_id, label) in zip(axes, order):
        sub = d[d.DOCUMENT_ID == doc_id].sort_values("IDX").reset_index(drop=True)
        n = len(sub)
        slot = 100.0 / n
        width = max(slot, 0.20)  # floor so long documents stay visible
        left = np.arange(n) / n * 100.0
        ax.add_patch(Rectangle((0, -0.42), 100, 3.3, facecolor=hs.PAPER, linewidth=0))
        for k, cid in enumerate(cids):
            row = 2 - k
            hits = sub[cid].to_numpy().astype(bool)
            ax.bar(
                left[hits],
                height=0.68,
                width=width,
                bottom=row - 0.34,
                color=COLORS[cid],
                linewidth=0,
                align="edge",
            )
            cnt, pct = int(hits.sum()), 100 * hits.sum() / n
            ax.text(
                101.4,
                row,
                f"{cnt:>3d}",
                va="center",
                ha="left",
                fontsize=12.0,
                weight="bold",
                color=text_colour(COLORS[cid]),
            )
            ax.text(
                105.8,
                row,
                f"{pct:4.1f}%",
                va="center",
                ha="left",
                fontsize=10.8,
                color=INK2,
            )
            ax.text(
                -1.2,
                row,
                LABELS[cid],
                va="center",
                ha="right",
                fontsize=10.3,
                color=text_colour(COLORS[cid]),
                weight="bold",
            )
        any_hit = sub[cids].to_numpy().any(axis=1)
        ax.text(
            -1.2,
            3.36,
            label,
            va="center",
            ha="right",
            fontsize=13.8,
            weight="bold",
            color=INK,
        )
        ax.text(
            101.4,
            3.36,
            f"{n} passages · {int(any_hit.sum())} carry a justice label "
            f"({100 * any_hit.mean():.1f}%)",
            va="center",
            ha="left",
            fontsize=9.8,
            color=INK2,
        )
        ax.set_xlim(-32, 116)
        ax.set_ylim(-0.75, 3.8)
        ax.axis("off")
        for tick in (0, 25, 50, 75, 100):
            ax.text(
                tick, -0.85, f"{tick}%", ha="center", va="top", fontsize=7.9, color=INK3
            )
    fig.subplots_adjust(top=0.845, bottom=0.115, hspace=0.45)
    titled(
        fig,
        title,
        how,
        "All passage types included, since the classifiers ran on every passage; passages under 20 characters are dropped. Short documents move several points per passage.",
    )
    save(fig, name)


NBSAPS = [
    ("UNFCCC.document.i00007868.n0000", "Uganda NBSAP  (2026)"),
    ("UNFCCC.document.i00007484.n0000", "Armenia NBSAP  (2026)"),
    ("UNFCCC.document.i00007864.n0000", "Portugal NBSAP  (2026)"),
]
PLEDGES = [
    ("UNFCCC.document.i00006565.n0000", "Türkiye NDC 3.0  (Nov 2025)"),
    ("UNFCCC.document.i00000391.n0000", "Türkiye LT-LEDS  (Nov 2024)"),
    ("UNFCCC.document.i00004795.n0000", "Australia 2035 NDC  (Sep 2025)"),
    ("UNFCCC.document.i00007760.n0000", "Australia LT-LEDS3  (Nov 2025)"),
]
HOW_TO_READ = (
    "Each tick is one passage carrying that label, positioned by where it falls in the document and drawn one\n"
    "passage wide — so equal ink means an equal share of the document."
)


def fig_deep_dive_nbsap() -> None:
    _deep_dive(
        NBSAPS,
        "05a_deep_dive_nbsaps",
        "Three biodiversity plans, three balances of justice",
        HOW_TO_READ,
        (14.6, 7.4),
    )


def fig_deep_dive_pledges() -> None:
    _deep_dive(
        PLEDGES,
        "05b_deep_dive_pledges",
        "Headline pledges talk about justice; the strategies behind them talk less",
        HOW_TO_READ,
        (14.6, 9.0),
    )


# ---------------------------------------------------------------- figure 6
def fig_laws() -> None:
    """Every Australian and Turkish law, by justice density."""
    lw = pd.read_parquet(DATA / "law_rates.parquet").copy()
    lw["pct"] = 100 * lw.ANY_JUSTICE / lw.PASSAGES_TOTAL
    lw = lw[lw.PASSAGES_TOTAL >= 30]
    # Colour means *country* here and nothing else, so it uses the two-colour
    # set kept clear of the classifier trio.
    cc = COUNTRY_COLORS

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.2, 8.4), gridspec_kw={"width_ratios": [2.35, 1]}
    )

    top = lw.sort_values("pct", ascending=True).tail(20)
    y = np.arange(len(top))
    ax.barh(y, top.pct, height=0.66, color=[cc[c] for c in top.COUNTRY], linewidth=0)
    xmax = top.pct.max() * 1.30
    for yi, (_, r) in zip(y, top.iterrows()):
        title = r.TITLE if len(r.TITLE) <= 52 else r.TITLE[:50] + "…"
        ax.text(-0.55, yi, title, va="center", ha="right", fontsize=10.3, color=INK)
        ax.text(
            r.pct + xmax * 0.012,
            yi,
            f"{r.pct:.1f}%",
            va="center",
            ha="left",
            fontsize=10.6,
            weight="bold",
            color=INK2,
        )
        ax.text(
            r.pct + xmax * 0.098,
            yi,
            f"{int(r.ANY_JUSTICE)}/{int(r.PASSAGES_TOTAL)}",
            va="center",
            ha="left",
            fontsize=8.6,
            color=INK3,
        )
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.8, len(top) - 0.2)
    ax.set_xlabel("% of passages carrying any justice label", fontsize=10.6, color=INK2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        "Top 20 laws by justice density",
        loc="left",
        fontsize=12.6,
        weight="bold",
        color=INK,
        pad=10,
    )

    agg = lw.groupby("COUNTRY").agg(
        laws=("ID", "size"),
        passages=("PASSAGES_TOTAL", "sum"),
        Q32=("Q32", "sum"),
        Q911=("Q911", "sum"),
        Q912=("Q912", "sum"),
        anyj=("ANY_JUSTICE", "sum"),
    )
    cids = ["Q32", "Q911", "Q912"]
    x = np.arange(len(cids))
    w = 0.36
    for i, country in enumerate(["Australia", "Türkiye"]):
        vals = [100 * agg.loc[country, k] / agg.loc[country, "passages"] for k in cids]
        ax2.bar(
            x + (i - 0.5) * w,
            vals,
            width=w * 0.88,
            color=text_colour(cc[country]),
            linewidth=0,
            label=country,
        )
        for xi, v, k in zip(x + (i - 0.5) * w, vals, cids):
            ax2.text(
                xi,
                v + 0.16,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11.3,
                weight="bold",
                color=INK2,
            )
            ax2.text(
                xi,
                v + 0.62,
                f"{int(agg.loc[country, k]):,}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color=INK3,
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[c].replace(" ", "\n") for c in cids], fontsize=10.1)
    ax2.set_ylim(0, 12.6)
    ax2.set_ylabel("% of all law passages", fontsize=10.6, color=INK2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=10.8, loc="upper right")
    ax2.set_title(
        "All laws pooled, by country",
        loc="left",
        fontsize=12.6,
        weight="bold",
        color=INK,
        pad=10,
    )
    fig.subplots_adjust(top=0.855, bottom=0.14, left=0.24, wspace=0.28)
    titled(
        fig,
        "Turkish climate law carries more justice language than Australian",
        "Left: the 20 laws with the highest share of justice-labelled passages. Right: all laws pooled by country.\nColour marks the country here, not the classifier.",
        "Laws with at least 30 passages. 'Law' is the front-end category; policies are excluded. Composition differs: Türkiye's set includes a national development plan.",
    )
    save(fig, "06_laws")


# ---------------------------------------------------------------- figure 7
def fig_country_words() -> None:
    """
    Australia vs Türkiye vocabulary, split by justice type.

    Pooled across all three classifiers the comparison mostly reproduced the
    distributive class, which is three times the size of the others. Run within
    each type it separates: Australia's procedural language is about First
    Nations and the states, Türkiye's is about participation and institutions.

    Per-row rate figures are dropped here — at small-multiple size they doubled
    the label height for information that is in results/country_log_odds.csv.
    """
    co = pd.read_csv(RESULTS / "country_log_odds.csv")
    cc = COUNTRY_COLORS
    n_uni, n_bi = 8, 3
    concepts = ["Q32", "Q911", "Q912"]

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 7.6))
    for ax, concept in zip(axes, concepts):
        sides = {}
        for country in ("Australia", "Türkiye"):
            sel = co[(co.concept == concept) & (co.country == country)]
            uni = sel[sel.ngram == 1].head(n_uni)
            bi = sel[sel.ngram == 2].head(n_bi)
            sides[country] = pd.concat([uni, bi], ignore_index=True)
        rows = list(range(n_uni)) + [r + 0.9 for r in range(n_uni, n_uni + n_bi)]
        y = np.array([max(rows) - r for r in rows])
        zmax = max(sides[c].z.abs().max() for c in sides)
        edge = zmax * 2.62
        label_x = zmax * 1.32
        gap = zmax * 0.018  # 2px-equivalent surface gap between the two fills

        ax.barh(
            y,
            -sides["Australia"].z.abs(),
            left=-gap,
            height=0.56,
            color=text_colour(cc["Australia"]),
            linewidth=0,
        )
        ax.barh(
            y,
            sides["Türkiye"].z.abs(),
            left=gap,
            height=0.56,
            color=text_colour(cc["Türkiye"]),
            linewidth=0,
        )
        for country, sign, ha in (("Australia", -1, "right"), ("Türkiye", 1, "left")):
            for yi, (_, r) in zip(y, sides[country].iterrows()):
                ax.text(
                    sign * (abs(r.z) + gap + zmax * 0.035),
                    yi,
                    f"{abs(r.z):.0f}",
                    va="center",
                    ha="left" if sign > 0 else "right",
                    fontsize=9.4,
                    weight="bold",
                    color=INK2,
                )
                ax.text(
                    sign * label_x,
                    yi,
                    r.term,
                    va="center",
                    ha=ha,
                    fontsize=11.0,
                    color=INK,
                )

        gap_y = max(rows) - n_uni + 0.45
        ax.axhline(gap_y, color=GRID, linewidth=0.8, xmin=0.06, xmax=0.94)
        n_aus = int(sides["Australia"].n_aus.iloc[0])
        n_tur = int(sides["Türkiye"].n_tur.iloc[0])
        ax.set_xlim(-edge, edge)
        ax.set_ylim(-1.15, max(y) + 1.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.text(
            0,
            max(y) + 1.05,
            LABELS[concept],
            ha="center",
            va="center",
            fontsize=14.4,
            weight="bold",
            color=text_colour(COLORS[concept]),
        )
        ax.text(
            -label_x,
            max(y) + 0.58,
            f"Australia · {n_aus:,}",
            ha="right",
            va="center",
            fontsize=10.3,
            weight="bold",
            color=text_colour(cc["Australia"]),
        )
        ax.text(
            label_x,
            max(y) + 0.58,
            f"Türkiye · {n_tur:,}",
            ha="left",
            va="center",
            fontsize=10.3,
            weight="bold",
            color=text_colour(cc["Türkiye"]),
        )
        ax.text(
            0,
            -0.95,
            "single words above the rule, two-word terms below",
            ha="center",
            va="center",
            fontsize=8.4,
            color=INK3,
            style="italic",
        )

    fig.subplots_adjust(top=0.855, bottom=0.13, wspace=0.10)
    titled(
        fig,
        "The same two countries sound different in each kind of justice",
        "Terms most over-represented in each country's justice passages, measured against the other country and run\nseparately within each classifier. Bar length is log-odds z; single words above the rule, two-word terms below.",
        "Multilateral fund projects excluded — Türkiye has 1,113 such passages and Australia none. Turkish documents are largely machine-translated.",
    )
    save(fig, "07_country_vocabulary")


# ---------------------------------------------------------------- figure 8
JUST_TRANSITION = [
    ("Q47", "just transition"),
    ("Q58", "social inclusion"),
    ("Q53", "social protection"),
    ("Q68", "decent work"),
    ("Q1754", "aligning skills"),
    ("Q69", "green jobs"),
    ("Q1744", "legal safeguards for vulnerable groups"),
]
IMPACTED_GROUPS = [
    ("Q704", "women and minority genders"),
    ("Q695", "youth"),
    ("Q676", "marginalized ethnicity"),
    ("Q684", "indigenous people"),
    ("Q1167", "people with limited assets"),
    ("Q690", "people with health conditions"),
    ("Q701", "people on the move"),
    ("Q708", "elderly people"),
    ("Q1016", "sexual minority"),
]


def fig_concept_overlap() -> None:
    """
    Which neighbouring concepts each justice classifier travels with.

    Lift, not raw co-occurrence: P(concept | justice) / P(concept). Raw rates
    would mostly restate how common each concept is, so a large concept like
    "women and minority genders" would dominate every column whether or not
    justice passages are actually enriched for it.
    """
    c = pd.read_parquet(DATA / "concept_cooccurrence.parquet").set_index("CONCEPT_ID")
    cids = ["Q32", "Q911", "Q912"]
    n_all = float(c.N_ALL.iloc[0])
    lift = {
        k: (c[f"WITH_{k}"] / float(c[f"N_{k}"].iloc[0])) / (c.N_CONCEPT / n_all)
        for k in cids
    }
    rate = {k: 100 * c[f"WITH_{k}"] / float(c[f"N_{k}"].iloc[0]) for k in cids}
    vmax = max(float(lift[k].max()) for k in cids)

    rows = (
        [("family", "Impacted groups  (Q672)")]
        + IMPACTED_GROUPS
        + [("family", "Just transition  (Q47)")]
        + JUST_TRANSITION
    )[::-1]

    fig, ax = plt.subplots(figsize=(12.8, 10.4))
    y = 0.0
    for cid, label in rows:
        if cid == "family":
            ax.text(
                -0.16,
                y + 0.30,
                label,
                ha="right",
                va="center",
                fontsize=13.2,
                weight=700,
                color=INK,
            )
            y += 0.75
            continue
        for j, k in enumerate(cids):
            v = float(lift[k].get(cid, np.nan))
            col = sequential(v, vmax)
            ax.add_patch(Rectangle((j, y), 0.96, 0.9, facecolor=col, linewidth=0))
            ax.text(
                j + 0.48,
                y + 0.58,
                f"{v:.1f}\u00d7",
                ha="center",
                va="center",
                fontsize=15.6,
                weight=700,
                color=ink_on(col),
            )
            ax.text(
                j + 0.48,
                y + 0.26,
                f"{float(rate[k].get(cid, np.nan)):.1f}% of its passages",
                ha="center",
                va="center",
                fontsize=7.9,
                color=ink_on(col),
                alpha=0.85,
            )
        ax.text(
            -0.16, y + 0.58, label, ha="right", va="center", fontsize=11.5, color=INK
        )
        ax.text(
            -0.16,
            y + 0.28,
            f"{cid} \u00b7 {int(c.N_CONCEPT.get(cid, 0)):,} passages",
            ha="right",
            va="center",
            fontsize=8.4,
            color=INK3,
        )
        y += 1.0

    for j, k in enumerate(cids):
        ax.text(
            j + 0.48,
            y + 0.36,
            LABELS[k],
            ha="center",
            va="bottom",
            fontsize=12.6,
            weight=700,
            color=text_colour(COLORS[k]),
        )
        ax.text(
            j + 0.48, y + 0.12, k, ha="center", va="bottom", fontsize=9.4, color=INK3
        )
    ax.set_xlim(-3.35, 3.05)
    ax.set_ylim(-0.25, y + 0.95)
    ax.axis("off")
    fig.subplots_adjust(top=0.86, bottom=0.10)
    titled(
        fig,
        "Procedural justice travels with people; distributive justice travels with no one",
        "Each cell is how much more often a concept appears in that classifier's passages than in the corpus at large.\nSmall figures are the raw share of that classifier's passages carrying the concept.",
        "Concept families taken from the concept store hierarchy; only members with a primary classifier appear. Q911 and Q912 are formally subconcepts of Q32, so some structure is ontology, not discourse.",
    )
    save(fig, "08_concept_overlap")


# ---------------------------------------------------------------- figure 9
MULTI = hs.NEUTRAL


def _stack(ax, sub, bands, x):
    ax.stackplot(
        x,
        [sub[b].to_numpy() for b, _, _ in bands],
        colors=[c for _, c, _ in bands],
        linewidth=0,
    )


def fig_timeline() -> None:
    """
    Justice language over time, in absolute passages, overall and by corpus.

    The aggregate share falls after 2022 while corpus composition changes
    underneath it. The per-corpus row exists to make that visible rather than
    letting the headline read as a decline in policy discourse.
    """
    t = pd.read_parquet(DATA / "justice_timeline.parquet")
    t = t[t.YR >= 2000]
    # Blue and Forest are the one weak pair for colour-vision deficiency, so
    # Mustard is stacked between them and they never share an edge.
    bands = [
        ("ONLY_Q911", Q911, "distributive only"),
        ("ONLY_Q912", Q912, "procedural only"),
        ("ONLY_Q32", Q32, "climate justice only"),
        ("MULTIPLE", MULTI, "more than one label"),
    ]
    cols = [b for b, _, _ in bands]

    tot = t.groupby("YR")[cols + ["PASSAGES_TOTAL"]].sum().sort_index()
    tot["JUSTICE"] = tot[cols].sum(axis=1)
    x = tot.index.to_numpy()

    fig = plt.figure(figsize=(15.4, 12.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.9, 1], hspace=0.30, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    _stack(ax, tot, bands, x)

    ax2 = ax.twinx()
    ax2.plot(
        x, tot.PASSAGES_TOTAL, color=INK, linewidth=2.0, linestyle=(0, (5, 2)), zorder=5
    )
    ax2.set_xlim(2000, 2030.5)
    # Deliberately not the same headroom as the left axis: with equal headroom
    # both series peak in 2024 at the same fractional height and the line
    # appears to trace the top of the stack.
    # Fixed maxima rather than data-driven headroom, so the corpus line sits
    # above the stack in every year: 450k/160k = 2.8 is below the smallest
    # total-to-justice ratio in the series (3.0, at the 2019 peak).
    ax2.set_ylim(0, 450_000)
    ax2.set_ylabel(
        "all passages published that year  (dashed line)", fontsize=10.8, color=INK
    )
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax2.spines[["top"]].set_visible(False)

    ax.set_ylim(0, 160_000)
    ax.set_ylabel(
        "passages carrying a justice label  (stacked areas)", fontsize=10.8, color=INK2
    )
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.set_xlim(2000, 2030.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)

    # Labelled on 2025, the last complete year.
    cum, anchors = 0, []
    for col, colour, lab in bands:
        v = int(tot[col].loc[2025])
        cum += v
        anchors.append([cum - v / 2, lab, colour, v])
    gap = tot.JUSTICE.max() * 0.085
    for i in range(1, len(anchors)):
        anchors[i][0] = max(anchors[i][0], anchors[i - 1][0] + gap)
    for ypos, lab, colour, v in anchors:
        ax.text(
            2026.4,
            ypos + gap * 0.16,
            lab,
            va="center",
            ha="left",
            fontsize=10.6,
            color=text_colour(colour),
            weight=700,
        )

    # Share against 2016, the Paris Agreement year, set beside the y-axis.
    def share(y):
        return 100 * tot.JUSTICE.loc[y] / tot.PASSAGES_TOTAL.loc[y]

    ax.text(
        0.013,
        0.93,
        f"{share(2016):.0f}% of all passages in 2016",
        transform=ax.transAxes,
        fontsize=12.0,
        weight=700,
        color=INK2,
    )
    ax.text(
        0.013,
        0.875,
        f"{share(2025):.0f}% in 2025",
        transform=ax.transAxes,
        fontsize=12.0,
        weight=700,
        color=INK2,
    )
    ax.set_title(
        "Combined corpora", loc="left", fontsize=13.8, weight=700, color=INK, pad=10
    )

    order = ["Law + Policy", "UN submission", "MCF project"]
    ymax = max(
        t[t.CORPUS_GROUP == g].groupby("YR")[cols].sum().sum(axis=1).max()
        for g in order
    )
    for i, grp in enumerate(order):
        axi = fig.add_subplot(gs[1, i])
        sub = t[t.CORPUS_GROUP == grp].groupby("YR")[cols + ["PASSAGES_TOTAL"]].sum()
        sub = sub.reindex(x, fill_value=0)
        _stack(axi, sub, bands, x)
        axi.set_ylim(0, ymax * 1.24)
        axi.set_xlim(2000, 2026)
        axi.spines[["top", "right"]].set_visible(False)
        axi.yaxis.grid(True, color=GRID, linewidth=0.7)
        axi.set_axisbelow(True)
        axi.tick_params(labelsize=7.5)
        axi.yaxis.set_major_formatter(lambda v, _: f"{int(v / 1000)}k" if v else "0")
        if i:
            axi.set_yticklabels([])
        j = sub[cols].sum(axis=1)
        share_now = 100 * j.loc[2025] / max(sub.PASSAGES_TOTAL.loc[2025], 1)
        share_then = 100 * j.loc[2016] / max(sub.PASSAGES_TOTAL.loc[2016], 1)
        axi.set_title(grp, loc="left", fontsize=11.5, weight=700, color=INK, pad=8)
        axi.text(
            0.02,
            0.92,
            f"{share_then:.0f}% in 2016  \u2192  {share_now:.0f}% in 2025",
            transform=axi.transAxes,
            fontsize=9.6,
            color=INK2,
            weight=700,
        )

    fig.subplots_adjust(top=0.85, bottom=0.125, left=0.065, right=0.885)
    titled(
        fig,
        "Justice language did not decline after 2022 \u2014 the corpus changed shape",
        "Stacked areas are passages carrying each label, by year of publication. The dashed line is the size of the whole\ncorpus that year, on an independent right-hand axis — the two do not share a vertical scale.",
        "Corporate disclosure is excluded: that dataset has not been updated this year. The remaining dip is composition — hold the 2016 corpus mix fixed and 2025 reads 37.2%, above 2016's own 33.1%. 2026 is a partial year.",
    )
    save(fig, "09_timeline")


def main() -> None:
    for fn in (
        fig_vocabulary,
        fig_overlap,
        fig_corpus_heatmap,
        fig_regions,
        fig_deep_dive_nbsap,
        fig_deep_dive_pledges,
        fig_laws,
        fig_country_words,
        fig_concept_overlap,
        fig_timeline,
    ):
        fn()


if __name__ == "__main__":
    main()
