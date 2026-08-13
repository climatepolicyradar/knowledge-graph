# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
#
# The pandas and scipy stubs do not narrow well here: pyright reads a Series
# as an ndarray (losing .str/.head) and a sparse matrix as a scalar (losing
# .toarray()). Every such call is exercised on each run, so the suppression is
# scoped to these analysis scripts rather than applied repo-wide.
"""
Word-count statistics separating the three justice classifiers.

Two complementary measures, both plain word counts:

1. TF-IDF over three class-documents (each class's passages concatenated).
   Simple and familiar, but on merged documents it rewards terms that are rare
   overall, so it drifts toward oddities.

2. Log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn
   2008), z-scored. This is the headline measure. It asks "is this word used at
   a different rate in class A than in the comparison set, more than sampling
   noise would explain", with the background corpus supplying the prior so rare
   words are shrunk rather than amplified.

Both run on the *exclusive* passages of each class — passages where exactly one
of the three classifiers fired — which is the sharpest read on what separates
them from each other.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

CLASSES = {
    "only_Q32": "Climate justice (Q32)",
    "only_Q911": "Distributive justice (Q911)",
    "only_Q912": "Procedural justice (Q912)",
}

# Boilerplate that survives stopword removal and swamps policy text without
# saying anything about justice. Kept deliberately short — the point is to see
# the domain language, not to hand-tune the result.
DOMAIN_STOPWORDS = {
    "shall",
    "may",
    "must",
    "will",
    "would",
    "should",
    "article",
    "paragraph",
    "section",
    "annex",
    "table",
    "figure",
    "page",
    "chapter",
    "pursuant",
    "thereof",
    "herein",
    "said",
    "et",
    "al",
    "e",
    "g",
    "ie",
    "eg",
    "www",
    "http",
    "https",
    "org",
    "com",
    "pdf",
    "doc",
    "pp",
    "vol",
    "no",
    "nos",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | DOMAIN_STOPWORDS)

TOKEN_PATTERN = r"(?u)\b[a-z][a-z-]{2,}\b"  # letters only, 3+ chars, keeps hyphens


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "text_samples.parquet")
    df["CONTENT"] = (
        df["CONTENT"].str.lower().map(lambda s: re.sub(r"\s+", " ", s).strip())
    )
    # Policy corpora repeat boilerplate verbatim across documents; a repeated
    # passage is one piece of language, not evidence of prevalence.
    before = len(df)
    df = df.drop_duplicates(subset=["GRP", "CONTENT"])
    print(f"deduplicated {before:,} -> {len(df):,} passages")
    return df


def tfidf_top(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Classic TF-IDF, treating each class as one merged document."""
    docs = [" ".join(df.loc[df.GRP == c, "CONTENT"]) for c in CLASSES]
    vec = TfidfVectorizer(
        stop_words=STOPWORDS,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    matrix = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for i, cls in enumerate(CLASSES):
        scores = matrix[i].toarray().ravel()
        for rank, j in enumerate(np.argsort(-scores)[:n], start=1):
            rows.append(
                {"cls": cls, "rank": rank, "term": vocab[j], "tfidf": scores[j]}
            )
    return pd.DataFrame(rows)


def _log_odds_scores(y1: np.ndarray, y2: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Monroe, Colaresi & Quinn (2008) eq. 16-22, z-scored."""
    n1, n2, alpha0 = y1.sum(), y2.sum(), alpha.sum()
    delta = np.log((y1 + alpha) / (n1 + alpha0 - y1 - alpha)) - np.log(
        (y2 + alpha) / (n2 + alpha0 - y2 - alpha)
    )
    return delta / np.sqrt(1.0 / (y1 + alpha) + 1.0 / (y2 + alpha))


def log_odds(df: pd.DataFrame, n_uni: int = 16, n_bi: int = 10) -> pd.DataFrame:
    """
    Log-odds ratio with informative Dirichlet prior, per class vs the rest.

    Unigrams and bigrams are ranked in *separate* vocabularies rather than one
    pooled one. A bigram is necessarily rarer than either of its parts, so in a
    pooled ranking the prior shrinks it harder and almost no bigrams survive
    into the top slots — which is why "human rights" and "first nations" were
    invisible next to "rights" and "nations". Ranking them apart lets each
    compete against terms of its own frequency regime.

    Comparison set is the other two justice classes, so the result answers
    "what marks this classifier out from its siblings" rather than "what marks
    justice language out from policy language".
    """
    subset = df[df.GRP.isin(CLASSES)]
    rows = []
    for ngram, keep in ((1, n_uni), (2, n_bi)):
        vec = CountVectorizer(
            stop_words=STOPWORDS,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(ngram, ngram),
            min_df=25,
        )
        counts = vec.fit_transform(subset["CONTENT"])
        vocab = np.array(vec.get_feature_names_out())
        grp = subset["GRP"].to_numpy()

        # Prior from the background corpus, scaled so it carries the weight of
        # a modest pseudo-sample rather than dominating.
        bg = (
            np.asarray(
                vec.transform(df.loc[df.GRP == "background", "CONTENT"]).sum(axis=0)
            )
            .ravel()
            .astype(float)
        )
        alpha = bg / max(bg.sum(), 1) * 1000.0 + 0.01

        per_class = {
            c: np.asarray(counts[grp == c].sum(axis=0)).ravel().astype(float)
            for c in CLASSES
        }
        for cls in CLASSES:
            y1 = per_class[cls]
            y2 = np.sum([v for k, v in per_class.items() if k != cls], axis=0)
            z = _log_odds_scores(y1, y2, alpha)
            n1, n2 = y1.sum(), y2.sum()
            for rank, j in enumerate(np.argsort(-z)[:keep], start=1):
                rows.append(
                    {
                        "cls": cls,
                        "ngram": ngram,
                        "rank": rank,
                        "term": vocab[j],
                        "z": z[j],
                        "count_in_class": int(y1[j]),
                        "count_in_others": int(y2[j]),
                        "rate_per_10k_in_class": 1e4 * y1[j] / n1,
                        "rate_per_10k_in_others": 1e4 * y2[j] / n2,
                    }
                )
    return pd.DataFrame(rows)


def country_log_odds(n_uni: int = 8, n_bi: int = 3) -> pd.DataFrame:
    """
    Australia vs Türkiye, run separately within each justice type.

    Multilateral Climate Fund projects are dropped: Türkiye has 1,113 such
    passages and Australia none, so leaving them in would make this partly a
    comparison of MCF reporting language against everything else.

    Turkish documents in this corpus are largely machine-translated, so some of
    what separates them is translation register rather than policy substance.
    """
    df = pd.read_parquet(DATA / "country_justice_text.parquet")
    df = df[df.CATEGORY != "Multilateral Climate Fund project"].copy()
    df["CONTENT"] = (
        df["CONTENT"].str.lower().map(lambda s: re.sub(r"\s+", " ", s).strip())
    )
    df = df.drop_duplicates(subset=["COUNTRY", "CONTENT"])

    # Country self-references separate the two sets perfectly and say nothing.
    country_words = [
        "australia",
        "australian",
        "australians",
        "aus",
        "türkiye",
        "turkiye",
        "turkey",
        "turkish",
        "tur",
        "republic",
    ]
    rows = []
    for concept in ("Q32", "Q911", "Q912"):
        sub = df[df[concept] == 1]
        print(f"  {concept}: {sub.COUNTRY.value_counts().to_dict()}")
        for ngram, keep in ((1, n_uni), (2, n_bi)):
            vec = CountVectorizer(
                stop_words=STOPWORDS + country_words,
                token_pattern=TOKEN_PATTERN,
                ngram_range=(ngram, ngram),
                min_df=5,
            )
            counts = vec.fit_transform(sub["CONTENT"])
            vocab = np.array(vec.get_feature_names_out())
            side = sub["COUNTRY"].to_numpy()
            y1 = (
                np.asarray(counts[side == "Australia"].sum(axis=0))
                .ravel()
                .astype(float)
            )
            y2 = np.asarray(counts[side == "Türkiye"].sum(axis=0)).ravel().astype(float)
            n1, n2 = y1.sum(), y2.sum()
            alpha = (y1 + y2) / (n1 + n2) * 500.0 + 0.01
            z = _log_odds_scores(y1, y2, alpha)
            order = np.concatenate([np.argsort(-z)[:keep], np.argsort(z)[:keep]])
            for k, j in enumerate(order):
                rows.append(
                    {
                        "concept": concept,
                        "ngram": ngram,
                        "country": "Australia" if k < keep else "Türkiye",
                        "rank": (k % keep) + 1,
                        "term": vocab[j],
                        "z": z[j],
                        "n_aus": int((side == "Australia").sum()),
                        "n_tur": int((side == "Türkiye").sum()),
                        "rate_per_10k_aus": 1e4 * y1[j] / n1,
                        "rate_per_10k_tur": 1e4 * y2[j] / n2,
                    }
                )
    return pd.DataFrame(rows)


def justice_vs_background(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """What separates any-justice language from the rest of the corpus."""
    vec = CountVectorizer(
        stop_words=STOPWORDS,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=25,
    )
    df2 = df.assign(side=np.where(df.GRP == "background", "background", "justice"))
    counts = vec.fit_transform(df2["CONTENT"])
    vocab = np.array(vec.get_feature_names_out())
    side = df2["side"].to_numpy()
    y1 = np.asarray(counts[side == "justice"].sum(axis=0)).ravel().astype(float)
    y2 = np.asarray(counts[side == "background"].sum(axis=0)).ravel().astype(float)
    n1, n2 = y1.sum(), y2.sum()
    alpha = (y1 + y2) / (n1 + n2) * 1000.0 + 0.01
    alpha0 = alpha.sum()
    delta = np.log((y1 + alpha) / (n1 + alpha0 - y1 - alpha)) - np.log(
        (y2 + alpha) / (n2 + alpha0 - y2 - alpha)
    )
    z = delta / np.sqrt(1.0 / (y1 + alpha) + 1.0 / (y2 + alpha))
    order = np.argsort(-z)[:n]
    return pd.DataFrame(
        {
            "rank": range(1, len(order) + 1),
            "term": vocab[order],
            "z": z[order],
            "rate_per_10k_justice": 1e4 * y1[order] / n1,
            "rate_per_10k_background": 1e4 * y2[order] / n2,
        }
    )


def main() -> None:
    df = load()
    print("\npassages per class after dedup:")
    print(df.GRP.value_counts().to_string())

    co = country_log_odds()
    co.to_csv(OUT / "country_log_odds.csv", index=False)

    lo = log_odds(df)
    lo.to_csv(OUT / "log_odds_by_class.csv", index=False)
    tf = tfidf_top(df)
    tf.to_csv(OUT / "tfidf_by_class.csv", index=False)
    jb = justice_vs_background(df)
    jb.to_csv(OUT / "justice_vs_background.csv", index=False)

    for cls, label in CLASSES.items():
        print(f"\n=== {label} — top distinctive terms (log-odds z) ===")
        for ngram, tag in ((1, "unigrams"), (2, "bigrams")):
            sub = lo[(lo.cls == cls) & (lo.ngram == ngram)].head(8)
            print(
                f"  [{tag}] "
                + ", ".join(f"{r['term']} ({r['z']:.0f})" for _, r in sub.iterrows())
            )

    print("\n=== TF-IDF cross-check (top 12) ===")
    for cls, label in CLASSES.items():
        terms = tf[tf.cls == cls].head(12)["term"].tolist()
        print(f"  {label}: {', '.join(terms)}")

    print("\n=== Australia vs Türkiye, by justice type ===")
    for concept in ("Q32", "Q911", "Q912"):
        print(f"  {concept}:")
        for country in ("Australia", "Türkiye"):
            for ngram, tag in ((1, "uni"), (2, "bi")):
                sel = co[
                    (co.concept == concept)
                    & (co.country == country)
                    & (co.ngram == ngram)
                ]
                print(
                    f"    {country:<10s} [{tag}] "
                    + ", ".join(
                        f"{r['term']} ({abs(r['z']):.0f})" for _, r in sel.iterrows()
                    )
                )

    print("\n=== Any justice vs background corpus ===")
    for _, r in jb.head(15).iterrows():
        print(
            f"  {r['term']:<32s} z={r['z']:6.1f}  "
            f"{r['rate_per_10k_justice']:6.1f} vs {r['rate_per_10k_background']:5.1f} /10k"
        )


if __name__ == "__main__":
    main()
