"""
Just transition charts for the COP31 ILO Just Transition Pavilion deck.

Reads the CSVs in data/ (pulled with the SQL in queries/ on 11 Nov 2026) and writes
PNG + SVG + plotted CSV into figures/.

Charts build on Topics (classifier output), not on wording:
- "just transition Topic": the Q47 classifier. Topics are hierarchical, so every passage
  tagged with a sub-topic (social protection, social inclusion, decent work, aligning skills,
  green jobs, legal safeguards) is also tagged Q47 and counts as just transition content.
- "generic mention": tagged Q47 but none of the six sub-topics, i.e. reached only through
  the root's own labels ("just transition", "leave no one behind", ...).
- "keyword search": concept-store labels matched in SQL for concepts inside the tree that
  have no classifier (social dialogue & trade unions, transition-industry workers). Marked
  with an asterisk and "keyword search, not Topic".
The explicit term ("just / fair / equitable transition", a text search) is used only where
the point is to compare wording with Topics: charts 2b and 7.

Run from the repo root:  uv run python demos/just_transition_cop31/charts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / ".claude" / "skills" / "cpr-charts"))
import cpr_style as cpr  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "figures"
ACCESSED = "11 Nov 2026"
SOURCE = "Climate Policy Radar"


def pct(x: float) -> str:
    return f"{x:.0%}"


def flags() -> pd.DataFrame:
    return pd.read_csv(DATA / "03_topic_flags.csv")


def label_notes(ax, y, notes, s):
    """Grey second line under each y tick label."""
    for yi, note in zip(y, notes):
        if note:
            ax.annotate(
                note,
                (0, yi),
                xycoords=("axes fraction", "data"),
                xytext=(-s["body"] * 0.8, -s["body"] * 0.62),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=s["small"],
                color=cpr.INK3,
            )


def keyword_note(ax, value_text, s):
    """Grey 'keyword search, not Topic' after a value label that already ends in '*'."""
    ax.annotate(
        "keyword search, not Topic",
        (1, 0.5),
        xycoords=value_text,
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=s["small"],
        color=cpr.INK3,
    )


def bar_frame(ax):
    ax.xaxis.set_visible(False)
    cpr.frame(ax, grid=None)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(cpr.GRID)


# ---------------------------------------------------------------------------------
# 1. Just transition Topic share of passages, by year and corpus (small multiples)
# ---------------------------------------------------------------------------------
def chart_by_year() -> None:
    s = cpr.apply("slide")
    d = flags()
    d = d[d.section == "year"].rename(columns={"grp": "corpus", "yr": "year"})[
        ["corpus", "year", "passages", "jt"]
    ]
    d["year"] = d.year.astype(int)
    d["share"] = d.jt / d.passages
    order = [
        "UNFCCC submissions",
        "Policies",
        "Laws",
        "Climate fund projects",
        "Court documents",
    ]

    fig, axes = cpr.figure(ncols=len(order), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, name in zip(axes, order):
        g = d[d.corpus == name].sort_values("year")
        ax.fill_between(
            g.year, 0, g.share, color=cpr.LIGHT_BLUE, lw=0, zorder=2, alpha=0.6
        )
        ax.plot(g.year, g.share, color=cpr.INKY_BLUE, lw=2, zorder=3)
        last = g.iloc[-1]
        ax.scatter([last.year], [last.share], s=18, color=cpr.INKY_BLUE, zorder=4)
        falling = (
            g.share.iloc[-2] > last.share
        )  # label below the point so the line doesn't cross it
        ax.annotate(
            f"{last.share:.1%}",
            (last.year, last.share),
            xytext=(0, -8 if falling else 7),
            textcoords="offset points",
            ha="center",
            va="top" if falling else "bottom",
            fontsize=s["body"],
            weight="semibold",
            color=cpr.INKY_NAVY,
        )
        cpr.frame(ax, grid="y")
        ticks = [2013, 2017, 2021, 2025]
        ax.set_xticks(ticks, [f"'{str(t)[2:]}" for t in ticks])
        ax.set_xlim(2011.5, 2025.9)
        ax.set_ylim(0, 0.16)
        ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        cpr.panel_title(
            ax,
            name,
            f"n={g.passages.sum() / 1e6:.1f}m passages"
            if g.passages.sum() >= 1e6
            else f"n={g.passages.sum() / 1e3:.0f}k passages",
        )

    cpr.titled(
        fig,
        "Just transition content is rising in UN submissions and climate fund projects",
        "What share of each year’s passages carry a Just Transition Topic?",
        cpr.footer(
            f"{d.passages.sum() / 1e6:.1f} million paragraphs and list items, 2012–2025, by publication year. "
            "UNFCCC submissions exclude inventories; nature-only laws and policies excluded.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "01_jt_by_year", OUT, dpi=300, data=d)


# ---------------------------------------------------------------------------------
# 2b. NDC rounds: the term vs the just transition Topic (paired bars) - a wording comparison
# ---------------------------------------------------------------------------------
def chart_ndc_rounds_topic() -> None:
    s = cpr.apply("slide_half")
    c = pd.read_csv(DATA / "02_ndc_rounds.csv")
    rows = []
    for r, label in [
        (1, "First NDCs\n2015–19"),
        (2, "Updated NDCs\n2020–24"),
        (3, "NDC 3.0\n2025–26"),
    ]:
        have = c[c[f"ndc{r}"] == 1]
        rows.append(
            {
                "round": label.replace("\n", " "),
                "label": f"{label}\nn={len(have)}",
                "countries": len(have),
                "topic": int(have[f"ndc{r}_topic"].sum()),
                "term": int(have[f"ndc{r}_jt"].sum()),
            }
        )
    d = pd.DataFrame(rows)
    d["share_topic"] = d.topic / d.countries
    d["share_term"] = d.term / d.countries

    fig, ax = cpr.figure()
    x = np.arange(len(d))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, d.share_topic, width=w, color=cpr.CONTEXT, zorder=2)
    ax.bar(x + w / 2 + 0.01, d.share_term, width=w, color=cpr.HIGHLIGHT, zorder=2)
    for xi, row in zip(x, d.itertuples()):
        ax.text(
            xi - w / 2 - 0.01,
            row.share_topic + 0.02,
            pct(row.share_topic),
            ha="center",
            va="bottom",
            fontsize=s["body"],
            weight="semibold",
            color=cpr.text_colour(cpr.CONTEXT),
        )
        ax.text(
            xi + w / 2 + 0.01,
            row.share_term + 0.02,
            pct(row.share_term),
            ha="center",
            va="bottom",
            fontsize=s["body"],
            weight="semibold",
            color=cpr.INKY_BLUE,
        )
    ax.set_xticks(x, d.label, color=cpr.INK)
    for tick in ax.get_xticklabels():
        tick.set_linespacing(1.25)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_visible(False)
    cpr.frame(ax, grid=None)

    cpr.titled(
        fig,
        [
            ("Just transition ", cpr.INK),
            ("topics", cpr.text_colour(cpr.CONTEXT)),
            (" were in NDCs long before the ", cpr.INK),
            ("term", cpr.INKY_BLUE),
        ],
        "Share of countries whose NDC carries the just transition topic (classifier) or uses the term (text "
        "search), by NDC round (rounds by publication year).",
        cpr.footer(
            "Countries with an NDC in our database, all files pooled.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "02b_ndc_rounds_topic", OUT, dpi=300, data=d.drop(columns="label"))


# ---------------------------------------------------------------------------------
# 3. Just transition sub-topics in NDC 3.0s (ranked bars)
# ---------------------------------------------------------------------------------
# (column, label, validated classifier?, note on nesting)
NDC_TOPICS = [
    ("inclusion", "Social inclusion", True, ""),
    ("socprot", "Social protection", True, ""),
    ("decent", "Decent work", True, ""),
    ("skills", "Aligning skills", True, ""),
    ("greenjobs", "Green jobs", True, ""),
    ("legal", "Legal safeguards for vulnerable groups", True, ""),
    (
        "dialogue",
        "Social dialogue & trade unions",
        False,
        "part of decent work and social inclusion",
    ),
    (
        "transition_workers",
        "Protecting transition-industry workers",
        False,
        "part of social protection",
    ),
    (
        "informal",
        "Informal economy",
        True,
        "sector Topic, outside the just transition tree",
    ),
]


def chart_ndc_topics() -> None:
    s = cpr.apply("slide")
    n3 = flags()
    n3 = n3[n3.section == "ndc3"]
    d = pd.DataFrame(
        [
            {
                "topic": label,
                "column": col,
                "validated_classifier": clf,
                "note": note,
                "countries": len(n3),
                "with_topic": int(n3[col].sum()),
            }
            for col, label, clf, note in NDC_TOPICS
        ]
    )
    d["share"] = d.with_topic / d.countries
    d = d.sort_values("share", ascending=False).reset_index(drop=True)

    fig, ax = cpr.figure()
    y = np.arange(len(d)) * 1.0
    for yi, row in zip(y, d.itertuples()):
        ax.barh(yi, row.share, height=0.62, color=cpr.HIGHLIGHT, zorder=2)
        value = ax.text(
            row.share + 0.01,
            yi,
            pct(row.share) + ("*" if not row.validated_classifier else ""),
            va="center",
            fontsize=s["body"],
            weight="semibold",
            color=cpr.INKY_BLUE,
        )
        if not row.validated_classifier:
            keyword_note(ax, value, s)
    ax.set_yticks(
        y,
        [f"{lab}\n" if note else lab for lab, note in zip(d.topic, d.note)],
        color=cpr.INK,
    )
    label_notes(ax, y, d.note, s)
    ax.set_ylim(y[-1] + 0.6, -0.6)
    ax.set_xlim(0, 1.0)
    bar_frame(ax)

    cpr.titled(
        fig,
        "Most NDC 3.0s cover social inclusion and protection; few mention social dialogue",
        "What share of NDC 3.0s mention each Just Transition Topic?",
        cpr.footer(
            f"{len(n3)} countries with a 2025–26 NDC in our database; all files pooled per country.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "03_ndc_topics", OUT, dpi=300, data=d)


# ---------------------------------------------------------------------------------
# 4. Work vs energy in just transition content (grouped bars)
# ---------------------------------------------------------------------------------
# (column, label, validated classifier?, group, bold summary row?)
WORK_ENERGY = [
    ("any_work", "Any work sub-topic", True, "work", True),
    ("decent", "Decent work", True, "work", False),
    ("skills", "Aligning skills", True, "work", False),
    ("greenjobs", "Green jobs", True, "work", False),
    ("dialogue", "Social dialogue & trade unions", False, "work", False),
    ("any_energy", "Any energy Topic", True, "energy", True),
    ("supplymit", "Energy supply mitigation", True, "energy", False),
    ("renew", "of which renewable energy", True, "energy", False),
    ("esector", "Energy supply sector", True, "energy", False),
    ("fossil", "Fossil fuels", True, "energy", False),
    ("coal", "of which coal", True, "energy", False),
    ("electrif", "Electrification", True, "energy", False),
]


def chart_energy_vs_work() -> None:
    s = cpr.apply("slide")
    t = flags()
    allp = t[(t.section == "corpus") & (t.grp == "All")].iloc[0].copy()
    allp["any_work"] = allp.g_work + allp.g_both
    n = int(allp.jt)
    d = pd.DataFrame(
        [
            {
                "topic": label,
                "column": col,
                "validated_classifier": clf,
                "group": grp,
                "summary": summ,
                "jt_passages": n,
                "with_topic": int(allp[col]),
            }
            for col, label, clf, grp, summ in WORK_ENERGY
        ]
    )
    d["share"] = d.with_topic / d.jt_passages
    colour = {"energy": cpr.CONTEXT, "work": cpr.HIGHLIGHT}

    fig, ax = cpr.figure()
    y, ypos = 0.0, []
    for i, row in enumerate(d.itertuples()):
        if i and row.group != d.group.iloc[i - 1]:
            y += 0.6  # gap between the groups
        ypos.append(y)
        y += 1
    y = np.array(ypos)
    for yi, row in zip(y, d.itertuples()):
        ax.barh(yi, row.share, height=0.66, color=colour[row.group], zorder=2)
        value = ax.text(
            row.share + 0.004,
            yi,
            pct(row.share) + ("*" if not row.validated_classifier else ""),
            va="center",
            fontsize=s["body"],
            weight="semibold" if row.summary else "normal",
            color=cpr.INKY_BLUE
            if row.group == "work"
            else cpr.text_colour(cpr.CONTEXT),
        )
        if not row.validated_classifier:
            keyword_note(ax, value, s)
    ax.set_yticks(y, d.topic, color=cpr.INK)
    for tick, summ in zip(ax.get_yticklabels(), d.summary):
        if summ:
            tick.set_fontweight("semibold")
    ax.set_ylim(y[-1] + 0.6, -0.6)
    ax.set_xlim(0, 0.32)
    bar_frame(ax)

    cpr.titled(
        fig,
        [
            ("Just transition content mentions ", cpr.INK),
            ("work", cpr.INKY_BLUE),
            (" about twice as often as ", cpr.INK),
            ("energy", cpr.text_colour(cpr.CONTEXT)),
        ],
        "Of passages carrying a Just Transition Topic, what share are about work, and what share also mention "
        "an energy Topic?",
        cpr.footer(
            f"{n:,} passages with a Just Transition Topic in laws, policies, UNFCCC submissions, climate fund "
            "projects and court documents, 2020–2025.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "04_energy_vs_work", OUT, dpi=300, data=d)


# ---------------------------------------------------------------------------------
# 5. Sector mix of just transition content vs the corpus (dumbbell)
# ---------------------------------------------------------------------------------
def chart_sector_mix() -> None:
    s = cpr.apply("slide")
    raw = pd.read_csv(DATA / "05_sector_lift.csv")
    # Expected sector mentions: each corpus's own sector rate applied to its JT passages,
    # so the corpus mix of JT passages is held fixed. Shares are of all sector mentions,
    # which divides out how many sectors a passage mentions at all.
    raw["expected"] = raw.passages_sector / raw.passages * raw.jt_passages
    d = raw.groupby("sector", as_index=False)[["jt_sector", "expected"]].sum()
    d["share_jt"] = d.jt_sector / d.jt_sector.sum()
    d["share_expected"] = d.expected / d.expected.sum()
    d["ratio"] = d.share_jt / d.share_expected
    d = d.sort_values("ratio", ascending=False).reset_index(drop=True)
    d["label"] = d.sector.replace(
        {"Extractive": "Mining, oil and gas", "ICT": "Information and communication"}
    )
    up = d.ratio >= 1.5
    down = d.ratio <= 0.67

    fig, ax = cpr.figure()
    y = np.arange(len(d))
    ax.hlines(
        y,
        d[["share_jt", "share_expected"]].min(axis=1),
        d[["share_jt", "share_expected"]].max(axis=1),
        color=cpr.BACKDROP,
        lw=3,
        zorder=1,
    )
    ax.scatter(
        d.share_expected,
        y,
        s=40,
        color=cpr.CONTEXT,
        zorder=3,
        edgecolor=cpr.SURFACE,
        lw=1,
    )
    ax.scatter(
        d.share_jt, y, s=46, color=cpr.HIGHLIGHT, zorder=4, edgecolor=cpr.SURFACE, lw=1
    )
    for yi, row, u, dn in zip(y, d.itertuples(), up, down):
        ax.text(
            1.01,
            yi,
            f"×{row.ratio:.1f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=s["small"],
            weight="semibold" if (u or dn) else "normal",
            color=cpr.INKY_BLUE if u else (cpr.INK if dn else cpr.INK3),
        )
    ax.set_yticks(y, d.label, color=cpr.INK, fontsize=s["small"])
    ax.set_ylim(len(d) - 0.5, -1.4)
    ax.set_xlim(0, 0.17)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cpr.frame(ax, grid="x")
    ax.text(
        0.0,
        -1.1,
        "● just transition content",
        color=cpr.INKY_BLUE,
        fontsize=s["small"],
        weight="semibold",
        va="center",
    )
    ax.text(
        0.075,
        -1.1,
        "● expected from the same corpora",
        color=cpr.text_colour(cpr.CONTEXT),
        fontsize=s["small"],
        weight="semibold",
        va="center",
    )
    ax.text(
        1.01,
        -1.1,
        "ratio",
        transform=ax.get_yaxis_transform(),
        va="center",
        fontsize=s["small"],
        color=cpr.INK3,
    )

    n_jt = int(raw.groupby("corpus").jt_passages.first().sum())
    cpr.titled(
        fig,
        "Just transition content leans on education and health, and under-weights energy and transport",
        "When passages with a Just Transition Topic mention an economic sector, which sectors come up more or less "
        "often than their corpora would predict?",
        cpr.footer(
            f"{int(d.jt_sector.sum()):,} sector mentions in {n_jt:,} passages with a Just Transition Topic, "
            "five corpora, 2020–2025.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "05_sector_mix", OUT, dpi=300, data=d.drop(columns="label"))


# ---------------------------------------------------------------------------------
# 6. Just transition Topic share of each region's passages, split by sub-topic
# ---------------------------------------------------------------------------------
# (column, legend label, colour)
SEGMENTS = [
    ("c_work", "work (decent work, skills, green jobs)", cpr.MUSTARD),
    ("c_protect", "social protection", cpr.INKY_BLUE),
    ("c_include", "social inclusion & safeguards", cpr.MID_BLUE),
    ("c_multiple", "multiple", cpr.LIGHT_BLUE),
    ("g_generic", "generic mention", cpr.CONTEXT),
]


def chart_regions() -> None:
    s = cpr.apply("slide")
    r = flags()
    r = r[r.section == "region"].copy()
    r["region"] = r.grp.str.split(" / ").str[0]
    a = r.groupby("region")[["passages", "jt"] + [c for c, _, _ in SEGMENTS]].sum()
    a.loc["All regions"] = a.sum()
    a["share"] = a.jt / a.passages
    for col, _, _ in SEGMENTS:
        a[col + "_share"] = a[col] / a.passages
        a[col + "_of_jt"] = a[col] / a.jt
    order = ["All regions"] + a.drop("All regions").sort_values(
        "share", ascending=False
    ).index.tolist()
    d = a.loc[order].reset_index(names="region")

    fig, ax = cpr.figure()
    y = np.arange(len(d)) * 1.0
    y[1:] += 0.4  # gap under the total
    left = np.zeros(len(d))
    for col, _, colour in SEGMENTS:
        ax.barh(
            y,
            d[col + "_share"],
            left=left,
            height=0.66,
            color=colour,
            edgecolor=cpr.SURFACE,
            lw=1,
            zorder=2,
        )
        left += d[col + "_share"]
    for yi, row in zip(y, d.itertuples()):
        ax.text(
            row.share + 0.0006,
            yi,
            f"{row.share:.1%}",
            va="center",
            fontsize=s["body"],
            weight="semibold",
            color=cpr.INK,
        )
    ax.set_yticks(y, d.region, color=cpr.INK)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.set_ylim(y[-1] + 0.6, -1.2)
    ax.set_xlim(0, 0.064)
    bar_frame(ax)
    # key: a square in each segment's fill, then its label in a readable ink
    x0 = 0.0
    for _, label, colour in SEGMENTS:
        sq = ax.text(x0, -0.95, "■", color=colour, fontsize=s["body"] + 2, va="center")
        fig.canvas.draw()
        x1 = sq.get_window_extent().transformed(ax.transData.inverted()).x1
        lab = ax.text(
            x1 + 0.0004,
            -0.95,
            label,
            color=cpr.INK2,
            fontsize=s["small"],
            weight="semibold",
            va="center",
        )
        fig.canvas.draw()
        x0 = lab.get_window_extent().transformed(ax.transData.inverted()).x1 + 0.002
    cpr.titled(
        fig,
        "South Asia and Sub-Saharan Africa documents mention just transition the most",
        "What share of each region’s passages carry a Just Transition Topic, and which of its sub-topics?",
        cpr.footer(
            f"{int(a.loc['All regions', 'passages']):,} paragraphs and list items in national climate laws and "
            "policies (nature-only excluded) and UNFCCC submissions, 2020–2025, tagged with one region.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "06_regions", OUT, dpi=300, data=d)


# ---------------------------------------------------------------------------------
# 7. Term frequency vs Topics elsewhere in the document - a wording comparison
# ---------------------------------------------------------------------------------
# (column, label, group)
TERM_VS = [
    ("child_any", "Any just transition sub-topic", "jt"),
    ("child_skills", "Aligning skills", "jt"),
    ("child_greenjobs", "Green jobs", "jt"),
    ("child_socprot", "Social protection", "jt"),
    ("child_decent", "Decent work", "jt"),
    ("child_legal", "Legal safeguards (rare)", "jt"),
    ("child_inclusion", "Social inclusion", "jt"),
    ("ctl_netzero", "Net-zero targets", "other"),
    ("ctl_fossil", "Fossil fuels", "other"),
    ("ctl_renew", "Renewable energy", "other"),
    ("ctl_ews", "Early warning systems", "other"),
    ("ctl_women", "Women & minority genders", "other"),
    ("ctl_adapt", "Adaptation", "other"),
    ("ctl_drought", "Drought", "other"),
]


def chart_term_vs_topics() -> None:
    s = cpr.apply("slide")
    raw = pd.read_csv(DATA / "07_term_vs_topics.csv")
    u = raw[
        (raw.section == "unfccc_doc")
        & raw.yr.between(2020, 2025)
        & (raw.passages >= 30)
    ].copy()
    u["group"] = np.select([u.jt == 0, u.jt >= 3], ["never", "often"], "sometimes")
    rows = []
    for col, label, grp in TERM_VS:
        share = u[col] / u.nonjt
        # Compare within document type: each document's share relative to its type's average,
        # so the mix of NDCs, BTRs, national communications etc. doesn't drive the ratio.
        rel = share / share.groupby(u.entity_type).transform("mean").replace(0, np.nan)
        m = rel.groupby(u.group).mean()
        raw_m = share.groupby(u.group).mean()
        rows.append(
            {
                "topic": label,
                "column": col,
                "group": grp,
                "ratio": m["often"] / m["never"],
                "ratio_without_type_adjustment": raw_m["often"] / raw_m["never"],
                "mean_share_never": raw_m["never"],
                "mean_share_often": raw_m["often"],
                "docs_never": int((u.group == "never").sum()),
                "docs_often": int((u.group == "often").sum()),
            }
        )
    d = pd.DataFrame(rows)
    d = pd.concat(
        [
            d[d.column == "child_any"],
            d[(d.group == "jt") & (d.column != "child_any")].sort_values(
                "ratio", ascending=False
            ),
            d[d.group == "other"].sort_values("ratio", ascending=False),
        ],
        ignore_index=True,
    )
    colour = {"jt": cpr.HIGHLIGHT, "other": cpr.CONTEXT}
    text = {"jt": cpr.INKY_BLUE, "other": cpr.text_colour(cpr.CONTEXT)}

    fig, ax = cpr.figure()
    y, ypos = 0.0, []
    for i, row in enumerate(d.itertuples()):
        if i and row.group != d.group.iloc[i - 1]:
            y += 0.6
        ypos.append(y)
        y += 1
    y = np.array(ypos)

    def fill(row):  # net-zero is the one comparison topic the title calls out
        return cpr.MUSTARD if row.column == "ctl_netzero" else colour[row.group]

    def ink(row):
        return (
            cpr.text_colour(cpr.MUSTARD)
            if row.column == "ctl_netzero"
            else text[row.group]
        )

    ax.barh(y, d.ratio, height=0.66, color=[fill(r) for r in d.itertuples()], zorder=2)
    ax.axvline(1, color=cpr.INK3, lw=0.8, ls=(0, (2, 2)), zorder=3)
    for yi, row in zip(y, d.itertuples()):
        ax.text(
            row.ratio + 0.03,
            yi,
            f"×{row.ratio:.1f}",
            va="center",
            fontsize=s["body"],
            weight="semibold"
            if row.column in ("child_any", "ctl_netzero")
            else "normal",
            color=ink(row),
        )
    ax.set_yticks(y, d.topic, color=cpr.INK)
    ax.get_yticklabels()[0].set_fontweight("semibold")
    ax.text(
        1,
        y[-1] + 0.55,
        "no difference",
        ha="center",
        va="top",
        fontsize=s["small"],
        color=cpr.INK3,
    )
    ax.set_ylim(y[-1] + 1.0, -0.6)
    ax.set_xlim(0, 2.3)
    bar_frame(ax)

    n_never, n_often = d.docs_never.iloc[0], d.docs_often.iloc[0]
    cpr.titled(
        fig,
        [
            ("Submissions that often say “just transition” cover its ", cpr.INK),
            ("sub-topics", cpr.INKY_BLUE),
            (" more, and ", cpr.INK),
            ("net-zero", cpr.MUSTARD),
            (" too", cpr.INK),
        ],
        "Relative to documents which never explicitly mention Just Transition, how much more likely are Topics "
        "to also occur anywhere in the same document? Ratio of mean Topic shares, 3+ mentions vs none.",
        cpr.footer(
            f"{len(u):,} UNFCCC submissions, 2020–25: {n_often} use the term 3+ times, {n_never} never. "
            "Topic shares exclude term passages and are scaled within document type.",
            accessed=ACCESSED,
            source=SOURCE,
        ),
    )
    cpr.save(fig, "07_term_vs_topics", OUT, dpi=300, data=d)


if __name__ == "__main__":
    chart_by_year()
    chart_ndc_rounds_topic()
    chart_ndc_topics()
    chart_energy_vs_work()
    chart_sector_mix()
    chart_regions()
    chart_term_vs_topics()
