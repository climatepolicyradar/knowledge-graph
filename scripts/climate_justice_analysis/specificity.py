# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
#
# The pandas and scipy stubs do not narrow well here: pyright reads a Series
# as an ndarray (losing .str/.head) and a sparse matrix as a scalar (losing
# .toarray()). Every such call is exercised on each run, so the suppression is
# scoped to these analysis scripts rather than applied repo-wide.
"""
How much explicit justice vocabulary does each classifier's output contain?

Reading samples by hand suggested the distributive-justice classifier fires on
a lot of ordinary mitigation prose — energy strategies, agriculture laws — with
no visible distributive content. This puts a number on that intuition instead
of trusting five hand-picked passages.

The probe is deliberately crude: four hand-built lexicons, and the share of
passages in each class containing at least one term from each. A passage can
be genuinely about distributive justice without using any of these words (that
is the whole premise of the classifiers, and the blog post says so explicitly),
so a low rate is *not* proof of a false positive. What it does give is a
like-for-like comparison across the three classifiers against a fixed yardstick,
plus the corpus background rate as a floor.
"""

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

LEXICONS = {
    "Explicit justice / equity": [
        "justice",
        "equity",
        "equitable",
        "equitably",
        "fair",
        "fairness",
        "unfair",
        "inequality",
        "inequalities",
        "inequitable",
        "just transition",
        "injustice",
    ],
    "Distributive markers": [
        "distribution",
        "distributive",
        "redistribut",
        "allocat",  # prefix: allocate/allocation/allocated
        "revenue",
        "subsidy",
        "subsidies",
        "compensat",
        "burden",
        "benefit-sharing",
        "benefit sharing",
        "affordab",
        "low-income",
        "low income",
        "poverty",
        "the poor",
        "disadvantaged",
        "cost of living",
        "energy poverty",
    ],
    "Procedural markers": [
        "consultation",
        "consult",
        "participat",  # prefix: participate/participation/participatory
        "stakeholder",
        "engagement",
        "consent",
        "transparen",
        "accountab",
        "grievance",
        "representation",
        "public hearing",
        "deliberat",
        "co-design",
        "inclusive decision",
    ],
    "Recognition markers": [
        "indigenous",
        "traditional knowledge",
        "customary",
        "cultural",
        "women",
        "gender",
        "youth",
        "marginali",
        "vulnerable",
        "minorit",
        "disabilit",
        "ethnic",
        "tribal",
        "first nations",
    ],
}

CLASS_LABELS = {
    "only_Q32": "Climate justice\n(Q32 only)",
    "only_Q911": "Distributive justice\n(Q911 only)",
    "only_Q912": "Procedural justice\n(Q912 only)",
    "all_three": "All three\nclassifiers",
    "background": "Corpus background\n(no justice label)",
}

PATTERNS = {
    name: re.compile("|".join(re.escape(t) for t in terms))
    for name, terms in LEXICONS.items()
}


def main() -> None:
    df = pd.read_parquet(DATA / "text_samples.parquet")
    df["CONTENT"] = df["CONTENT"].str.lower()
    df = df.drop_duplicates(subset=["GRP", "CONTENT"])

    for name, pat in PATTERNS.items():
        df[name] = df["CONTENT"].str.contains(pat, regex=True)
    df["Any of the four"] = df[list(LEXICONS)].any(axis=1)

    cols = list(LEXICONS) + ["Any of the four"]
    rates = (df.groupby("GRP")[cols].mean() * 100).round(1)
    rates = rates.reindex([c for c in CLASS_LABELS if c in rates.index])
    rates.index = [CLASS_LABELS[i] for i in rates.index]
    counts = (
        df.groupby("GRP")
        .size()
        .reindex([c for c in CLASS_LABELS if c in df.GRP.unique()])
    )
    rates["n passages"] = counts.values

    rates.to_csv(OUT / "lexicon_specificity.csv")
    pd.set_option("display.width", 200)
    print("Share of passages containing at least one term from each lexicon (%)\n")
    print(rates.to_string())

    print("\n\nShare with NO explicit justice vocabulary at all:")
    for grp, label in CLASS_LABELS.items():
        sub = df[df.GRP == grp]
        if len(sub):
            pct = 100 * (~sub["Any of the four"]).mean()
            print(
                f"  {label.replace(chr(10), ' '):<38s} {pct:5.1f}%   (n={len(sub):,})"
            )


if __name__ == "__main__":
    main()
