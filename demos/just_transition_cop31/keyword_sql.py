r"""
Build Snowflake regex predicates that mirror KeywordClassifier matching.

They cover concepts that have no classifier of their own (workers, social dialogue), using
the labels in data/concepts.json.

Matching mirrors knowledge_graph/classifier/keyword.py:
- a label with any uppercase or non-ASCII character matches case-sensitively, all
  others case-insensitively;
- separators are flexible: the label is split on whitespace/hyphens/dashes and rejoined
  with [[:space:]-–—]+;
- whole-word matching via \\b (Snowflake has no lookarounds; \\b is equivalent for labels
  that start and end with a word character, which all of these do).
None of the subtrees used here carries negative labels, so no suppression step is needed.

Because the search is a union ("does the passage match any label"), a label that contains
another label of the same search as a whole-word run is redundant and is dropped. That
changes the size of the regex, not which passages match.

Usage:  python keyword_sql.py   -> prints the predicates and label counts
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONCEPTS = json.loads((HERE / "data" / "concepts.json").read_text())

SEARCHES = {
    # Impacted workers (Q1160: workers, employees, working class, informal workers,
    # workers impacted by the transition, high-risk occupations) sits under impacted
    # groups, outside Q47. Protecting transition industry workers (Q1799) sits inside
    # Q47, under social protection (Q53) and workforce for a just transition (Q65).
    "workers": ["Q1160", "Q1799"],
    # Protecting transition industry workers alone (inside Q47, under social protection Q53).
    "transition_workers": ["Q1799"],
    # Social dialogue (Q54, under social inclusion Q58 and decent work Q68) and trade
    # union (Q59, under decent work Q68). Both are already inside the Q68 classifier.
    "social_dialogue": ["Q54", "Q59"],
}


def subtree(qid: str, acc: list[str] | None = None) -> list[str]:
    acc = [] if acc is None else acc
    if qid in acc or qid not in CONCEPTS:
        return acc
    acc.append(qid)
    for child in CONCEPTS[qid]["has_subconcept"]:
        subtree(child, acc)
    return acc


def labels_for(roots: list[str]) -> tuple[list[str], list[str]]:
    qids: list[str] = []
    for r in roots:
        subtree(r, qids)
    labels: set[str] = set()
    for q in qids:
        labels |= {CONCEPTS[q]["label"], *CONCEPTS[q]["alternative_labels"]}
        assert not CONCEPTS[q]["negative_labels"], (
            f"{q} has negative labels: handle them"
        )
    return qids, sorted(labels)


def case_sensitive(label: str) -> bool:
    return any(ch.isupper() or ord(ch) > 127 for ch in label)


def tokens(label: str) -> list[str]:
    return [t for t in re.split(r"[\s\-–—]+", label) if t]


def prune(labels: list[str]) -> list[str]:
    """Drop labels that contain another (case-insensitive) label as a whole-token run."""
    norm = {lab: " " + " ".join(tokens(lab.lower())) + " " for lab in labels}
    short = [lab for lab in labels if not case_sensitive(lab)]
    keep = []
    for lab in labels:
        n = norm[lab]
        if any(o != lab and norm[o] != n and norm[o] in n for o in short):
            continue
        keep.append(lab)
    # identical normalised forms: keep one
    seen, out = set(), []
    for lab in keep:
        key = (norm[lab], case_sensitive(lab))
        if key not in seen:
            seen.add(key)
            out.append(lab)
    return out


def to_regex(label: str) -> str:
    parts = [re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", t) for t in tokens(label)]
    return r"[[:space:]\-–—]+".join(parts)


SEP = r"[[:space:]\-–—]+"


def esc(tok: str) -> str:
    return re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", tok)


def trie_regex(labels: list[str], lower: bool) -> str:
    """One regex for many labels, sharing common leading tokens (same matches, shorter)."""
    root: dict = {}
    for lab in labels:
        node = root
        for tok in tokens(lab.lower() if lower else lab):
            node = node.setdefault(tok, {})
        node[""] = {}

    def build(node: dict) -> str:
        alts = []
        for tok in sorted(k for k in node if k):
            child = node[tok]
            rest = {k: v for k, v in child.items() if k}
            if not rest:
                alts.append(esc(tok))
            elif "" in child:
                alts.append(f"{esc(tok)}({SEP}{build(rest)})?")
            else:
                alts.append(f"{esc(tok)}{SEP}{build(rest)}")
        return alts[0] if len(alts) == 1 else "(" + "|".join(alts) + ")"

    return build(root)


def predicate(name: str, col: str = "p.content") -> str:
    _, labels = labels_for(SEARCHES[name])
    labels = prune(labels)
    ci = [lab for lab in labels if not case_sensitive(lab)]
    cs = [lab for lab in labels if case_sensitive(lab)]
    parts = []
    for group, flag in ((ci, "i"), (cs, "c")):
        if group:
            rx = r"\b(" + trie_regex(group, lower=(flag == "i")) + r")\b"
            rx = rx.replace("\\", "\\\\").replace("'", "''")  # SQL string literal
            parts.append(f"regexp_instr({col}, '{rx}', 1, 1, 0, '{flag}') > 0")
    return "(" + " or ".join(parts) + ")"


if __name__ == "__main__":
    for name, roots in SEARCHES.items():
        qids, labels = labels_for(roots)
        pruned = prune(labels)
        print(
            f"{name}: {len(qids)} concepts {qids}, {len(labels)} labels, {len(pruned)} after pruning"
        )
        print("  kept:", pruned)
        print("  sql length:", len(predicate(name)))
