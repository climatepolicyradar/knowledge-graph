"""
Pull the concept-store trees these charts depend on and save them to data/concepts.json.

Trees: Q47 just transition, Q1160 impacted workers (where "workers" and "working class"
live, outside Q47), plus the energy and mitigation concepts used in chart 4.
Prints each tree with its classifier status so the nesting (and any double counting) is
visible.

Run from the repo root:  uv run python demos/just_transition_cop31/concepts.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from knowledge_graph.wikibase import WikibaseSession  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOTS = sys.argv[1:] or ["Q47", "Q1160"]


def main() -> None:
    wb = WikibaseSession()
    seen: dict[str, dict] = {}

    def walk(qid: str) -> None:
        if qid in seen:
            return
        c = wb.get_concept(qid)
        seen[qid] = {
            "qid": qid,
            "label": c.preferred_label,
            "alternative_labels": sorted(c.alternative_labels),
            "negative_labels": sorted(c.negative_labels),
            "subconcept_of": list(c.subconcept_of),
            "has_subconcept": list(c.has_subconcept),
        }
        for child in c.has_subconcept:
            walk(child)

    for root in ROOTS:
        walk(root)

    out = HERE / "data" / "concepts.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing.update(seen)
    out.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"saved {len(seen)} concepts under {ROOTS} -> {out.name} ({len(existing)} total)"
    )


if __name__ == "__main__":
    main()
