"""Everything tunable about the experiment, and the console every module logs to."""

from rich.console import Console

from knowledge_graph.config import ensemble_metrics_dir

console = Console()

# The concepts from FUS-163, spanning easy/medium/hard. Q704 is deliberately absent:
# keywords and negative labels were added to it after its evaluation set was labelled,
# so it no longer reflects the concept the human labellers saw.
DEFAULT_CONCEPTS = [
    "Q618",
    "Q661",
    "Q2029",
    "Q47",
    "Q1277",
    "Q557",
    "Q1837",
    "Q912",
    "Q1829",
    "Q32",
]

FAMILY_A = "opus"

FAMILY_B = "gemini"

FAMILIES = {
    FAMILY_A: "openrouter:anthropic/claude-opus-5",
    FAMILY_B: "openrouter:google/gemini-3.1-pro-preview",
}

SEEDS = [1, 2, 3, 4, 5]

TEMPERATURE = 0.7

OPENROUTER_MODELS_PRICING_URL = "https://openrouter.ai/api/v1/models"

OPENROUTER_PRICING_CACHE_FILENAME = "model_pricing.json"

ESTIMATED_PROMPT_TOKENS_PER_PASSAGE = 4_300
"""Estimated prompt tokens in one classification call. An estimate, not a measurement.

Each call carries `DEFAULT_SYSTEM_PROMPT` formatted with the concept's markdown, plus the
passage. Over the ten FUS-163 concepts and their 2,015 gold passages that is a
passage-weighted mean of ~16,800 characters of system prompt and ~335 of passage text,
so ~17,100 characters ≈ 4,300 tokens.

The distribution behind that mean is very skewed: eight of the ten concepts sit between
2,300 and 6,400 characters, while `Q47` (2,982 alternative labels) and `Q557` (4,363) run
to 96,000 and 119,000. A single flat constant therefore overstates a small concept's cost
several-fold and understates those two.
"""

ESTIMATED_COMPLETION_TOKENS_PER_PASSAGE = 180
"""Estimated completion tokens in one classification call. An estimate, not a measurement."""

HEADLINE_SPAN_THRESHOLDS = [0, 0.25, 0.5]
"""The span-overlap thresholds the headline tables report, loosest first.

More than one, because a single threshold can't tell "the ensemble missed the mention"
apart from "it found the mention but drew the span differently": 0 credits any overlap at
all, and the drop from 0 to 0.5 is how much of the span-level score is boundary
disagreement rather than detection.
"""

SPAN_AGREEMENT_THRESHOLDS = HEADLINE_SPAN_THRESHOLDS + [0.9, 0.99]

PASSAGE_LEVEL = "passage"

HEADLINE_ENSEMBLE = "mixed_n5"

DEFAULT_OUTPUT_DIR = ensemble_metrics_dir / "eval_set_autolabelling"

AUTOMATE_F1 = 0.95
SEMI_AUTOMATE_F1 = 0.85
AUTOMATE_COVERAGE = 0.50

UNANIMOUS_POSITIVE_POLICY = "unanimous positive only"

UNANIMOUS_NEGATIVE_POLICY = "unanimous negative only"

AUTOMATE_VERDICT = "automate"
SEMI_AUTOMATE_VERDICT = "semi-automate"
BELOW_BAR_VERDICT = "below bar"

NOT_APPLICABLE = "n/a"
"""Rendered where precision/recall/F1 are undefined, so a sentinel 0.0 can't read as a score."""

COST_COLUMN = "cost_usd"
"""Estimated OpenRouter spend on one concept, in USD.

The per-concept tables' cost column, summed over that concept's passages. Its macro row
is a **mean**, like every other cell in that row — a row labelled "macro average" whose
cost cell was a sum read as though the ensemble were eight times cheaper than it is.

What the whole exercise costs is `TOTAL_COST_COLUMN` in the sizing table, where a total is
what the column is for and is named as such.

Any column with `usd` in its name gets currency formatting in `format_for_display`.
"""

TOTAL_COST_COLUMN = "total_cost_usd"
"""Estimated OpenRouter spend for one ensemble across every concept, in USD.

The sizing table's cost column, and a genuine total: it is the budget for running that
ensemble over the whole evaluation set, which is the number the sizing decision turns on.
Distinct from `COST_COLUMN` precisely so neither has to be read off a row label.
"""

AGREEMENT_CURVE_COLUMNS = [
    "ensemble",
    "n_members",
    "family",
    "disagreement_threshold",
    "n_passages",
    "macro_coverage",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_f1",
    "false_negatives",
    "n_human_remaining",
]

COMPLETENESS_COLUMNS = [
    "concept",
    "gold_passages",
    "members_expected",
    "members_cached",
    "members_complete",
    "worst_member_passages",
    "missing_members",
    "scored_passages",
    "excluded",
    "exclusion_reason",
]
"""The columns of `data_completeness.csv`, in reading order."""

POLICY_CONSOLE_COLUMNS = {
    "policy": "policy",
    "macro_coverage": "coverage",
    "macro_precision": "prec",
    "macro_recall": "rec",
    "macro_f1": "f1",
    "macro_npv": "npv",
    "false_negatives": "missed",
    "verdict": "verdict",
}

BY_ENSEMBLE_CONSOLE_COLUMNS = {
    "ensemble": "ensemble",
    "n_members": "n",
    "n_concepts": "concepts",
    "best_policy": "best policy",
    "macro_coverage": "coverage",
    "macro_precision": "prec",
    "macro_f1": "f1",
    "verdict": "verdict",
    TOTAL_COST_COLUMN: "USD total",
}

NEGATIVE_SIDE_CONSOLE_COLUMNS = {
    "concept": "concept",
    "predicted_negative": "no span",
    "negative_share": "share",
    "npv": "npv",
    "missed_mentions": "missed",
    "specificity": "spec",
}

VOTE_SPLIT_CONSOLE_COLUMNS = {
    "votes": "votes",
    "ensemble_label": "label",
    "n_passages": "n",
    "precision": "prec",
    "precision_95_ci": "prec 95% CI",
    "false_negatives": "missed",
    "missed_mention_rate": "missed rate",
}
