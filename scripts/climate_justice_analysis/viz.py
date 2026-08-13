# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "pandas>=2.2", "numpy>=1.26", "pyarrow>=16"]
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

Every figure carries its exact values as text so the whole thing can be rebuilt
in a design tool without going back to the data. Each is written to both PNG
(200 dpi, for review) and SVG (for redrawing).

Palette is the first three categorical slots of the reference data-viz palette
— blue / orange / aqua — which clear the all-pairs CVD and normal-vision gates
in light mode. Aqua sits below 3:1 against the surface, so the relief rule
applies: every mark carries a visible direct label.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8983"
GRID = "#e3e2de"

Q32, Q911, Q912 = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"Q32": Q32, "Q911": Q911, "Q912": Q912}
LABELS = {
    "Q32": "Climate justice",
    "Q911": "Distributive justice",
    "Q912": "Procedural justice",
}
# Sequential blue ramp, 100 -> 700, for magnitude encoding.
BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

mpl.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": 9,
        "axes.titlesize": 10,
        "svg.fonttype": "none",
    }
)


def sequential(value: float, vmax: float) -> str:
    """Pick a step from the blue ramp for a 0..vmax magnitude."""
    if vmax <= 0:
        return BLUE_RAMP[0]
    i = int(round((value / vmax) ** 0.75 * (len(BLUE_RAMP) - 1)))
    return BLUE_RAMP[max(0, min(len(BLUE_RAMP) - 1, i))]


def ink_on(hexcolor: str) -> str:
    r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


def titled(fig, title: str, subtitle: str, note: str = "") -> None:
    fig.text(0.012, 0.975, title, fontsize=15, weight="bold", color=INK, va="top")
    fig.text(0.012, 0.928, subtitle, fontsize=9.5, color=INK2, va="top")
    if note:
        fig.text(0.012, 0.017, note, fontsize=7.4, color=INK3, va="bottom")


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(
            FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE
        )
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .svg")


SOURCE = (
    "Source: Climate Policy Radar, PRODUCTION.PUBLISHED.PASSAGES (Snowflake), Aug 2026. "
    "Body text passages only (content_type='Text', >=20 chars), published non-principal documents."
)


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
    n_uni, n_bi = 12, 6
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
                fontsize=9.6,
                color=INK,
            )
            ax.text(
                -zmax * 0.045,
                yi - 0.22,
                f"{r.rate_per_10k_in_class:.0f} vs {r.rate_per_10k_in_others:.0f} per 10k",
                va="center",
                ha="right",
                fontsize=6.8,
                color=INK3,
            )
            ax.text(
                r.z + zmax * 0.02,
                yi,
                f"{r.z:.0f}",
                va="center",
                ha="left",
                fontsize=8.4,
                color=INK2,
                weight="bold",
            )
        gap_y = max(rows) - n_uni + 0.5
        ax.text(
            -zmax * 1.26,
            gap_y,
            "two-word terms, ranked separately",
            fontsize=7.2,
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
            color=COLORS[cid],
            weight="bold",
            fontsize=11.5,
            pad=16,
            loc="left",
        )
        ax.spines[:].set_visible(False)
        ax.text(
            -zmax * 1.26,
            -0.95,
            "bar length = log-odds z vs the other two classifiers",
            fontsize=7.2,
            color=INK3,
            va="center",
            ha="left",
        )
    fig.subplots_adjust(top=0.815, bottom=0.075, wspace=0.20)
    titled(
        fig,
        "Three justice classifiers, three vocabularies",
        "The 12 single words and 6 two-word terms most over-represented in the passages where each classifier fires alone, versus\n"
        "passages where only the other two fire. Grey figures under each term are its rate per 10,000 words in this class versus\n"
        "the other two combined. Distributive justice reads as mitigation vocabulary rather than allocation vocabulary — a finding\n"
        'about where distributive claims sit in this corpus, not a defect: the classifier was built so it would not need the word "equity".',
        "Log-odds ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), z-scored; prior from the unlabelled corpus. "
        "English stopwords plus legal boilerplate removed, min. 25 occurrences.\n"
        "One- and two-word terms are ranked in separate vocabularies: a two-word term is necessarily rarer than either of its parts, so "
        "in a pooled ranking the prior shrinks it harder and almost none survive.\n"
        + SOURCE
        + " Samples of 54,651 / 59,104 / 36,299 deduplicated exclusive passages.",
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
    bar_colors = [COLORS[c[0][0]] if len(c[0]) == 1 else "#9a998f" for c in combos]
    ax_bar.bar(x, heights, width=0.62, color=bar_colors, linewidth=0)
    for xi, (sets, n) in zip(x, combos):
        ax_bar.text(
            xi,
            n + total * 0.030,
            f"{n:,}",
            ha="center",
            va="bottom",
            fontsize=11.5,
            weight="bold",
            color=INK,
        )
        ax_bar.text(
            xi,
            n + total * 0.010,
            f"{100 * n / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=INK2,
            weight="bold",
        )
    ax_bar.set_ylim(0, max(heights) * 1.24)
    ax_bar.set_ylabel("passages", fontsize=9, color=INK2)
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
                color=COLORS[cid] if on else "#e3e2de",
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
            fontsize=10,
            weight="bold",
            color=COLORS[cid],
        )
        ax_set.text(
            tmax * 2.02,
            row + 0.02,
            f"{cid} · {totals[cid]:,} passages",
            ha="left",
            va="center",
            fontsize=7.6,
            color=INK3,
        )
        ax_set.text(
            tmax * 2.02,
            row - 0.24,
            f"{100 * totals[cid] / total:.0f}% of all justice passages",
            ha="left",
            va="center",
            fontsize=7,
            color=INK3,
        )
    ax_set.set_xlim(tmax * 2.10, 0)
    ax_set.axis("off")

    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.045, right=0.985)
    titled(
        fig,
        "Distributive justice does most of the work, and usually works alone",
        f"The seven ways the three classifiers can combine on a passage, across the {total:,} passages carrying at least one justice label.\n"
        "Bars are disjoint: every justice-labelled passage falls in exactly one of these seven columns, so they sum to 100%.\n"
        "Left-hand bars are each classifier's own total, which do overlap and so sum to more than 100%. Percentages above each bar are\n"
        "its share of all justice-labelled passages.",
        "Coloured bars are passages where a single classifier fired; grey bars are passages carrying more than one label.\n"
        + SOURCE,
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

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    for i, corpus in enumerate(rates.index):
        for j, cid in enumerate(cids):
            v = rates.loc[corpus, cid]
            col = sequential(v, float(rates[cid].max()))
            ax.add_patch(Rectangle((j, i), 0.96, 0.94, facecolor=col, linewidth=0))
            ax.text(
                j + 0.48,
                i + 0.47,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=16,
                weight="bold",
                color=ink_on(col),
            )
        ax.text(
            -0.16,
            i + 0.60,
            corpus,
            ha="right",
            va="center",
            fontsize=10,
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
            fontsize=7.6,
            color=INK3,
        )
    for j, cid in enumerate(cids):
        ax.text(
            j + 0.48,
            len(rates) + 0.30,
            LABELS[cid],
            ha="center",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color=COLORS[cid],
        )
        ax.text(
            j + 0.48,
            len(rates) + 0.14,
            f"{cid} · {baseline[cid]:.1f}% overall",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=INK3,
        )

    ax.set_xlim(-2.55, 3.05)
    ax.set_ylim(-0.25, len(rates) + 0.80)
    ax.axis("off")
    fig.subplots_adjust(top=0.78, bottom=0.10)
    titled(
        fig,
        "Funding documents talk about justice; laws and disclosures do not",
        "Share of each corpus's own body-text passages carrying each justice label, so corpus size is divided out.\n"
        "Shading runs light to dark with that percentage, scaled within each column — the three classifiers have very different\n"
        "base rates, given under each heading. Compare shades down a column, not across one.",
        'Litigation is omitted entirely: all three classifier specs carry dont_run_on: ["sabin"] and every published Litigation document '
        "is a Sabin record, so no inference was ever run there.\n" + SOURCE,
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
                color=COLORS[cid],
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
                    fontsize=8.4,
                    weight="bold",
                    color=INK2,
                )
        ax.set_xlim(0, xmax)
        ax.set_title(label, loc="left", fontsize=10.5, weight="bold", color=INK, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel(
            "% of body-text passages carrying the label",
            fontsize=8.8,
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
                fontsize=6.8,
                color=INK3,
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=10.2, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9.6,
        ncols=3,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.845),
    )

    ssa = [p[2].loc["Sub-Saharan Africa", "Q911"] for p in panels]
    eca = [p[2].loc["Europe & Central Asia", "Q911"] for p in panels]
    fig.subplots_adjust(top=0.755, bottom=0.13, left=0.16, wspace=0.07)
    titled(
        fig,
        "Removing the fund documents narrows the gap but does not close it",
        "Share of passages carrying each justice label, by World Bank region, before and after dropping Multilateral Climate Fund projects.\n"
        f"On distributive justice the Sub-Saharan Africa to Europe & Central Asia ratio moves only from {ssa[0] / eca[0]:.2f}× to {ssa[1] / eca[1]:.2f}×.\n"
        "Climate justice and procedural justice shrink much more, so fund reporting was inflating those two rather than the gradient as a whole.",
        "Litigation excluded from both panels (classifiers never ran on the Sabin corpus). Documents tagged to several regions are counted in "
        "each, so rows do not partition the corpus.\n"
        "The ranking also survives inside a single corpus: Sub-Saharan Africa leads Europe & Central Asia on distributive justice within UN "
        "submissions (25.2% vs 12.0%) and within Policy (38.9% vs 28.7%).\n" + SOURCE,
    )
    save(fig, "04_regions")


# ---------------------------------------------------------------- figure 5
def fig_deep_dive() -> None:
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
    order = [
        ("UNFCCC.document.i00007868.n0000", "Uganda NBSAP  (2026)"),
        ("UNFCCC.document.i00007484.n0000", "Armenia NBSAP  (2026)"),
        ("UNFCCC.document.i00007864.n0000", "Portugal NBSAP  (2026)"),
        ("UNFCCC.document.i00006565.n0000", "Türkiye NDC 3.0  (Nov 2025)"),
        ("UNFCCC.document.i00000391.n0000", "Türkiye LT-LEDS  (Nov 2024)"),
        ("UNFCCC.document.i00004795.n0000", "Australia 2035 NDC  (Sep 2025)"),
        ("UNFCCC.document.i00007760.n0000", "Australia LT-LEDS3  (Nov 2025)"),
    ]
    cids = ["Q32", "Q911", "Q912"]
    fig, axes = plt.subplots(len(order), 1, figsize=(13.8, 14.6))
    for ax, (doc_id, label) in zip(axes, order):
        sub = d[d.DOCUMENT_ID == doc_id].sort_values("IDX").reset_index(drop=True)
        n = len(sub)
        slot = 100.0 / n
        width = max(slot, 0.20)  # floor so long documents stay visible
        left = np.arange(n) / n * 100.0
        ax.add_patch(Rectangle((0, -0.42), 100, 3.3, facecolor="#f4f3f0", linewidth=0))
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
                fontsize=10,
                weight="bold",
                color=COLORS[cid],
            )
            ax.text(
                105.8,
                row,
                f"{pct:4.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color=INK2,
            )
            ax.text(
                -1.2,
                row,
                LABELS[cid],
                va="center",
                ha="right",
                fontsize=8.6,
                color=COLORS[cid],
                weight="bold",
            )
        any_hit = sub[cids].to_numpy().any(axis=1)
        ax.text(
            -1.2,
            3.36,
            label,
            va="center",
            ha="right",
            fontsize=11.5,
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
            fontsize=8.2,
            color=INK2,
        )
        ax.set_xlim(-32, 116)
        ax.set_ylim(-0.75, 3.8)
        ax.axis("off")
        for tick in (0, 25, 50, 75, 100):
            ax.text(
                tick, -0.85, f"{tick}%", ha="center", va="top", fontsize=6.6, color=INK3
            )
    fig.subplots_adjust(top=0.855, bottom=0.075, hspace=0.45)
    titled(
        fig,
        "Headline pledges talk about justice; the strategies behind them talk less",
        "Each tick is one passage carrying that label, positioned by where it falls in the document and drawn one passage wide, so equal\n"
        "ink means an equal share of the document. Both countries' current ANDCs are far denser in justice language than their own\n"
        "long-term strategies — Türkiye 51.9% against 27.9%, Australia 45.2% against 29.1% — and procedural justice is the thinnest\n"
        "strand nearly everywhere. Portugal's NBSAP is the exception: at 13.8% it is the most procedural document here bar one.",
        "All passage types included, not just body text, because the classifiers ran on every passage — much of the justice language in "
        "these documents sits in list items. Passages under 20 characters are dropped.\n"
        "The NDC-to-strategy gap is not purely rhetorical positioning: ANDCs are short political documents (54 and 155 passages here) while "
        "LT-LEDS are long technical ones (463 and 2,697), carrying inventory tables and sectoral detail that dilute any single theme. "
        "Türkiye's NDC 3.0 is 54 passages, so one passage there moves its rate by 1.9 points.\n"
        "DOCUMENTS.passage_count is not usable as a denominator here: it is rolled up from the v1 passages table and overstates the real v2 "
        "count 7-11x for these documents.\n"
        "Source: Climate Policy Radar, PRODUCTION.PUBLISHED.PASSAGES (Snowflake), Aug 2026.",
    )
    save(fig, "05_deep_dive_documents")


# ---------------------------------------------------------------- figure 6
def fig_laws() -> None:
    """Every Australian and Turkish law, by justice density."""
    lw_all = pd.read_parquet(DATA / "law_rates.parquet").copy()
    lw = lw_all.copy()
    lw["pct"] = 100 * lw.ANY_JUSTICE / lw.PASSAGES_TOTAL
    lw = lw[lw.PASSAGES_TOTAL >= 30]
    # 63 published AUS/TUR laws exist; law_rates only contains those with at
    # least one passage, and the 30-passage floor drops a few more.
    n_published, n_with_passages, n_shown = 63, len(lw_all), len(lw)
    # Colour means *country* in this figure and nothing else, so it deliberately
    # avoids the blue/orange/aqua the classifiers wear everywhere else. Violet
    # and yellow separate at CVD ΔE 41 and normal-vision ΔE 46.
    cc = {"Australia": "#4a3aa7", "Türkiye": "#eda100"}
    n_laws = lw.groupby("COUNTRY").size()

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.2, 8.4), gridspec_kw={"width_ratios": [2.35, 1]}
    )

    top = lw.sort_values("pct", ascending=True).tail(20)
    y = np.arange(len(top))
    ax.barh(y, top.pct, height=0.66, color=[cc[c] for c in top.COUNTRY], linewidth=0)
    xmax = top.pct.max() * 1.30
    for yi, (_, r) in zip(y, top.iterrows()):
        title = r.TITLE if len(r.TITLE) <= 52 else r.TITLE[:50] + "…"
        ax.text(-0.55, yi, title, va="center", ha="right", fontsize=8.6, color=INK)
        ax.text(
            r.pct + xmax * 0.012,
            yi,
            f"{r.pct:.1f}%",
            va="center",
            ha="left",
            fontsize=8.8,
            weight="bold",
            color=INK2,
        )
        ax.text(
            r.pct + xmax * 0.098,
            yi,
            f"{int(r.ANY_JUSTICE)}/{int(r.PASSAGES_TOTAL)}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=INK3,
        )
    ax.set_yticks([])
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.8, len(top) - 0.2)
    ax.set_xlabel("% of passages carrying any justice label", fontsize=8.8, color=INK2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(
        "Top 20 laws by justice density",
        loc="left",
        fontsize=10.5,
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
            color=cc[country],
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
                fontsize=9.4,
                weight="bold",
                color=INK2,
            )
            ax2.text(
                xi,
                v + 0.62,
                f"{int(agg.loc[country, k]):,}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK3,
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[c].replace(" ", "\n") for c in cids], fontsize=8.4)
    ax2.set_ylim(0, 12.6)
    ax2.set_ylabel("% of all law passages", fontsize=8.8, color=INK2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.set_title(
        "All laws pooled, by country",
        loc="left",
        fontsize=10.5,
        weight="bold",
        color=INK,
        pad=10,
    )
    summary = "   ·   ".join(
        f"{c}: {int(agg.loc[c, 'laws'])} laws, {int(agg.loc[c, 'passages']):,} passages, "
        f"{100 * agg.loc[c, 'anyj'] / agg.loc[c, 'passages']:.1f}% any justice"
        for c in ("Australia", "Türkiye")
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.24, wspace=0.28)
    ratio = (agg.loc["Türkiye", "anyj"] / agg.loc["Türkiye", "passages"]) / (
        agg.loc["Australia", "anyj"] / agg.loc["Australia", "passages"]
    )
    titled(
        fig,
        f"Turkish climate law carries {ratio:.0f}× Australia's justice density",
        f"Every published Australian and Turkish law in the database with at least 30 passages: "
        f"{int(n_laws['Australia'])} Australian, {int(n_laws['Türkiye'])} Turkish.\n"
        "Colour marks the country here, not the classifier. Small grey figures are labelled passages over total passages.\n"
        + summary,
        "Composition drives much of the country gap: Türkiye's set includes the Eleventh National Development Plan (924 passages, 27.5% "
        "justice-labelled), while Australia's largest law by volume is the Higher Education Support Act (3,526 passages, 4.1%).\n"
        f"'Law' is the front-end category; policies are excluded by request. Of {n_published} published laws, "
        f"{n_published - n_with_passages} have no passages at all and a further {n_with_passages - n_shown} fall below the "
        "30-passage floor.\n"
        "All passage types included (not just body text), matching the deep-dive panel; passages under 20 characters dropped.\n"
        "Source: Climate Policy Radar, PRODUCTION.PUBLISHED.PASSAGES (Snowflake), Aug 2026.",
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
    cc = {"Australia": "#4a3aa7", "Türkiye": "#eda100"}
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
            color=cc["Australia"],
            linewidth=0,
        )
        ax.barh(
            y,
            sides["Türkiye"].z.abs(),
            left=gap,
            height=0.56,
            color=cc["Türkiye"],
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
                    fontsize=7.8,
                    weight="bold",
                    color=INK2,
                )
                ax.text(
                    sign * label_x,
                    yi,
                    r.term,
                    va="center",
                    ha=ha,
                    fontsize=9.2,
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
            fontsize=12,
            weight="bold",
            color=COLORS[concept],
        )
        ax.text(
            -label_x,
            max(y) + 0.58,
            f"Australia · {n_aus:,}",
            ha="right",
            va="center",
            fontsize=8.6,
            weight="bold",
            color=cc["Australia"],
        )
        ax.text(
            label_x,
            max(y) + 0.58,
            f"Türkiye · {n_tur:,}",
            ha="left",
            va="center",
            fontsize=8.6,
            weight="bold",
            color=cc["Türkiye"],
        )
        ax.text(
            0,
            -0.95,
            "single words above the rule, two-word terms below",
            ha="center",
            va="center",
            fontsize=7,
            color=INK3,
            style="italic",
        )

    fig.subplots_adjust(top=0.78, bottom=0.13, wspace=0.10)
    titled(
        fig,
        "The same two countries sound different in each kind of justice",
        "Terms most over-represented in each country's justice-labelled passages, measured against the other country and run separately\n"
        "within each classifier. Bar length is log-odds z; the figure beside each bar is that z. Counts beside each country are the\n"
        "passages available in that panel. Australia's procedural language is First Nations and the states; Türkiye's is participation\n"
        "and institutions — a contrast the pooled comparison hid behind the much larger distributive class.",
        "Log-odds ratio with informative Dirichlet prior, z-scored, on justice-labelled body-text passages from published non-litigation "
        "documents tagged to exactly one of the two countries, deduplicated on passage text.\n"
        "Multilateral Climate Fund projects are excluded: Türkiye has 1,113 such passages and Australia none. Country names are removed "
        "from the vocabulary. One- and two-word terms are ranked separately.\n"
        "Procedural justice has the thinnest evidence of the three panels. Turkish documents here are largely machine-translated, so some "
        "separation is translation register rather than policy substance.\n"
        "Per-term rates per 10,000 words are in results/country_log_odds.csv.\n"
        + SOURCE,
    )
    save(fig, "07_country_vocabulary")


def main() -> None:
    for fn in (
        fig_vocabulary,
        fig_overlap,
        fig_corpus_heatmap,
        fig_regions,
        fig_deep_dive,
        fig_laws,
        fig_country_words,
    ):
        fn()


if __name__ == "__main__":
    main()
