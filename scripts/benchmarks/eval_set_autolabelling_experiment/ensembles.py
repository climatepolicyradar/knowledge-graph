"""What an ensemble is, what it costs, and how its predictions are run and cached."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from knowledge_graph.classifier.large_language_model import (
    DEFAULT_SYSTEM_PROMPT,
    LLMClassifier,
    LLMClassifierPrompt,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import label_passages_with_classifier
from knowledge_graph.operations.evaluate import create_gold_standard_labelled_passages
from knowledge_graph.operations.get_concept import get_concept_async
from scripts.benchmarks.eval_set_autolabelling_experiment.config import (
    ESTIMATED_COMPLETION_TOKENS_PER_PASSAGE,
    ESTIMATED_PROMPT_TOKENS_PER_PASSAGE,
    FAMILIES,
    FAMILY_A,
    FAMILY_B,
    NOT_APPLICABLE,
    OPENROUTER_MODELS_PRICING_URL,
    OPENROUTER_PRICING_CACHE_FILENAME,
    SEEDS,
    TEMPERATURE,
    console,
)

EnsembleMember = tuple[str, int]
"""A single ensemble member, identified by (family key, random seed)."""


@dataclass(frozen=True)
class NamedEnsemble:
    """A named ensemble, built from a subset of the cached members."""

    name: str
    members: tuple[EnsembleMember, ...]

    @property
    def size(self) -> int:
        """Number of classifiers in the ensemble."""
        return len(self.members)


def compose_possible_ensembles(max_members: int | None = None) -> list[NamedEnsemble]:
    """
    Build every ensemble we can compose from 5 members per family.

    Family-only ensembles answer "does stacking one model on itself help?"; mixed ones
    answer "is cross-family diversity worth the plumbing?". ``n1`` is the status quo: a
    single LLM classifier.

    Every ensemble is odd-sized, so `MajorityVote` can never tie and the whole
    experiment runs on one unambiguous decision rule. A balanced mixed ensemble is
    necessarily even, so the mixed splits are deliberately lopsided by one member: the
    price of avoiding ties is that one family gets the extra vote.

    :param max_members: drop ensembles larger than this. Answers "what is the smallest
        ensemble that can do the job?" — with a cap of 3, the analysis reports only what
        3 classifiers can achieve, so the verdict cannot lean on members you won't pay for.
    """

    ensembles = [
        NamedEnsemble(
            name=f"{family}_n{n}",
            members=tuple((family, seed) for seed in SEEDS[:n]),
        )
        for family in FAMILIES
        for n in (1, 3, 5)
    ]

    mixed_splits = [(2, 1), (3, 2), (4, 3), (5, 4)]
    ensembles.extend(
        NamedEnsemble(
            name=f"mixed_n{n_a + n_b}",
            members=tuple(
                [(FAMILY_A, seed) for seed in SEEDS[:n_a]]
                + [(FAMILY_B, seed) for seed in SEEDS[:n_b]]
            ),
        )
        for n_a, n_b in mixed_splits
    )

    if max_members is not None:
        ensembles = [ensemble for ensemble in ensembles if ensemble.size <= max_members]

    return ensembles


def family_of(ensemble_name: str) -> str:
    """The family an ensemble belongs to, read off its name."""
    return str(ensemble_name).split("_n")[0]


def all_members() -> list[EnsembleMember]:
    """
    Every member the experiment runs: one per (family, seed) pair.

    Deliberately the *full* set rather than the union of whichever ensembles survive
    ``--max-members``. It is the yardstick `analyse` filters concepts against, and that
    yardstick has to be independent of the cap: otherwise raising the cap would change
    which concepts are in scope, and two runs of the same data would not be comparable.
    """
    return [(family, seed) for family in FAMILIES for seed in SEEDS]


def describe_members(members: Iterable[EnsembleMember]) -> str:
    """Render members as ``family seed=N``, for a log line or a CSV cell."""
    return ", ".join(f"{family} seed={seed}" for family, seed in members)


@dataclass(frozen=True)
class TokenPrices:
    """Per-token USD prices for one model, as OpenRouter reports them."""

    prompt: float
    completion: float


def model_slug(model_name: str) -> str:
    """The bare OpenRouter model slug for one of `FAMILIES`' values."""
    return model_name.split(":", maxsplit=1)[-1]


def fetch_model_pricing() -> dict[str, TokenPrices]:
    """
    Fetch each family's per-token prices from OpenRouter, keyed by family.

    A family whose slug OpenRouter doesn't list is left out rather than defaulted, so that
    an unknown price stays distinguishable from a free one downstream.
    """

    response = httpx.get(OPENROUTER_MODELS_PRICING_URL, timeout=30.0)
    response.raise_for_status()

    pricing_by_slug = {
        str(model.get("id")): model.get("pricing") or {}
        for model in response.json().get("data", [])
    }

    prices: dict[str, TokenPrices] = {}
    for family, model_name in FAMILIES.items():
        pricing = pricing_by_slug.get(model_slug(model_name))
        if not pricing or "prompt" not in pricing or "completion" not in pricing:
            console.log(
                f"⚠️ OpenRouter lists no prompt/completion price for "
                f"{model_slug(model_name)} — `{family}` costs will read "
                f"`{NOT_APPLICABLE}`"
            )
            continue
        prices[family] = TokenPrices(
            prompt=float(pricing["prompt"]), completion=float(pricing["completion"])
        )

    return prices


def read_pricing_cache(path: Path) -> dict[str, TokenPrices]:
    """Read cached prices, returning nothing at all if the file can't be parsed."""

    try:
        cached = json.loads(path.read_text())
        return {
            family: TokenPrices(
                prompt=float(prices["prompt"]), completion=float(prices["completion"])
            )
            for family, prices in cached.items()
            if family in FAMILIES
        }
    except (OSError, ValueError, KeyError, TypeError) as e:
        console.log(f"⚠️ Couldn't read cached pricing from {path}: {e}")
        return {}


def write_pricing_cache(path: Path, prices: dict[str, TokenPrices]) -> None:
    """Cache the fetched prices, recording which slug each family was priced from."""

    path.write_text(
        json.dumps(
            {
                family: {
                    "model": model_slug(FAMILIES[family]),
                    "prompt": token_prices.prompt,
                    "completion": token_prices.completion,
                }
                for family, token_prices in prices.items()
            },
            indent=2,
        )
    )


def load_model_pricing(output_dir: Path) -> dict[str, TokenPrices]:
    """
    Per-token prices for every family, from the cache if it exists and the API if not.

    Fails soft: if there is no cache and the fetch doesn't work, this returns nothing and
    the cost columns are dropped rather than filled with zeros. A cost of $0 and an
    unknown cost are very different claims about an ensemble.
    """

    if (path := output_dir / OPENROUTER_PRICING_CACHE_FILENAME).exists():
        if cached := read_pricing_cache(path):
            return cached

    try:
        prices = fetch_model_pricing()
    except Exception as e:
        console.log(
            f"⚠️ Couldn't fetch OpenRouter pricing ({e}) and no usable cache at "
            f"{path} — costs will be omitted"
        )
        return {}

    if prices:
        write_pricing_cache(path, prices)
    return prices


def member_cost_per_passage(prices: TokenPrices) -> float:
    """What one classifier costs to run over one passage, in USD."""
    return (
        ESTIMATED_PROMPT_TOKENS_PER_PASSAGE * prices.prompt
        + ESTIMATED_COMPLETION_TOKENS_PER_PASSAGE * prices.completion
    )


def ensemble_cost_per_passage(
    ensemble: NamedEnsemble, prices_by_family: dict[str, TokenPrices]
) -> float:
    """
    What one ensemble costs per passage: the sum over its members, in USD.

    Every member sees every passage, so an ``n=5`` same-family ensemble is five times a
    single classifier and a mixed one is the sum of its members' families. No prompt
    caching is assumed — members differ only by seed, so a provider that did cache the
    shared system prompt across them would make this an over-estimate.

    Returns `nan` if any member's family has no price, so an unpriced ensemble renders as
    `NOT_APPLICABLE` rather than as a suspiciously cheap one.
    """

    if any(family not in prices_by_family for family, _ in ensemble.members):
        return math.nan

    return sum(
        member_cost_per_passage(prices_by_family[family])
        for family, _ in ensemble.members
    )


def ensemble_costs_per_passage(
    ensembles: Iterable[NamedEnsemble], prices_by_family: dict[str, TokenPrices]
) -> dict[str, float]:
    """Per-passage cost for every ensemble, keyed by ensemble name."""
    return {
        ensemble.name: ensemble_cost_per_passage(ensemble, prices_by_family)
        for ensemble in ensembles
    }


def concept_output_dir(output_dir: Path, wikibase_id: str) -> Path:
    """Return (and create) the directory holding one concept's cached predictions."""
    path = output_dir / str(wikibase_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def member_path(concept_dir: Path, member: EnsembleMember) -> Path:
    """Return the cache path for one ensemble member's predictions."""
    family, seed = member
    return concept_dir / f"member_{family}_seed{seed}.jsonl"


def write_passages(path: Path, passages: Iterable[LabelledPassage]) -> None:
    """
    Write labelled passages to a JSONL file, one per line.

    Writes to a temporary file and renames on success, so an interrupted run never
    leaves behind a truncated file that a later run would mistake for a complete one.
    """

    partial_path = path.with_suffix(".jsonl.partial")
    with open(partial_path, "w") as f:
        for passage in passages:
            f.write(passage.model_dump_json() + "\n")
    partial_path.rename(path)


def read_passages(path: Path) -> list[LabelledPassage]:
    """Read labelled passages from a JSONL file."""
    with open(path) as f:
        return [LabelledPassage.model_validate(json.loads(line)) for line in f if line]


def read_member_cache(
    concept_dir: Path, member: EnsembleMember
) -> dict[str, LabelledPassage]:
    """Load a member's already-predicted passages, keyed by passage id."""
    path = member_path(concept_dir, member)
    if not path.exists():
        return {}
    return {passage.id: passage for passage in read_passages(path)}


def load_cached_predictions(
    concept_dir: Path,
) -> (
    tuple[list[LabelledPassage], dict[EnsembleMember, dict[str, LabelledPassage]]]
    | None
):
    """
    Load one concept's gold passages and every cached member's predictions.

    Members are keyed by passage id rather than kept in list order: the gold set is
    refetched live on every `predict` run, so a cached member can hold a slightly
    different set of passages than the current gold file.
    """

    gold_path = concept_dir / "gold.jsonl"
    if not gold_path.exists():
        console.log(f"⚠️ No gold.jsonl in {concept_dir} — run `predict` first")
        return None

    gold = read_passages(gold_path)

    predictions_by_member: dict[EnsembleMember, dict[str, LabelledPassage]] = {}
    for family in FAMILIES:
        for seed in SEEDS:
            member: EnsembleMember = (family, seed)
            path = member_path(concept_dir, member)
            if not path.exists():
                continue
            predictions_by_member[member] = {p.id: p for p in read_passages(path)}

    if not predictions_by_member:
        console.log(f"⚠️ No cached members in {concept_dir}")
        return None

    return gold, predictions_by_member


UsableConcepts = dict[
    str, tuple[list[LabelledPassage], dict[EnsembleMember, dict[str, LabelledPassage]]]
]
"""Per concept, the gold passages to score on and every cached member's predictions."""


def load_usable_concepts(
    output_dir: Path, wikibase_ids: Iterable[str]
) -> tuple[UsableConcepts, list[dict[str, Any]]]:
    """
    Work out which concepts can be scored, and on which passages.

    Whether a concept is in scope is a property of the whole cache for that concept rather
    than of any one ensemble, so this runs once up front instead of per ensemble. A concept
    is excluded outright if a member was never run, and otherwise its gold set is
    restricted to the passages *every* member holds: `align_passages` drops a passage any
    member of that ensemble lacks, so without this a failed call would leave two ensembles
    within one concept scored on different passages.

    :return: the scorable concepts, and one completeness row per requested concept —
        including the excluded ones, since an absent row would be indistinguishable from a
        concept that was scored, which is the bug this filter exists to fix
    """

    expected_members = all_members()
    usable: UsableConcepts = {}
    completeness: list[dict[str, Any]] = []

    for wikibase_id in wikibase_ids:
        concept_dir = output_dir / wikibase_id
        loaded = load_cached_predictions(concept_dir)
        if loaded is None:
            completeness.append(
                {
                    "concept": wikibase_id,
                    "gold_passages": 0,
                    "members_expected": len(expected_members),
                    "members_cached": 0,
                    "members_complete": 0,
                    "worst_member_passages": 0,
                    "missing_members": describe_members(expected_members),
                    "scored_passages": 0,
                    "excluded": True,
                    "exclusion_reason": "nothing cached — run `predict`",
                }
            )
            continue
        gold, predictions_by_member = loaded

        console.log(
            f"{wikibase_id}: {len(gold)} gold passages, "
            f"{len(predictions_by_member)} cached members"
        )

        # a member holding fewer passages than gold had calls fail; those passages are
        # dropped from that member rather than scored as negatives
        passages_per_member = {
            member: len(predictions)
            for member, predictions in predictions_by_member.items()
        }
        incomplete = {
            member: count
            for member, count in passages_per_member.items()
            if count < len(gold)
        }
        for member, count in incomplete.items():
            console.log(
                f"  ⚠️ {member[0]} seed={member[1]} has {count}/{len(gold)} passages — "
                f"{len(gold) - count} failed and are excluded"
            )

        missing = [
            member for member in expected_members if member not in predictions_by_member
        ]
        common_ids = {
            passage.id
            for passage in gold
            if all(
                passage.id in predictions
                for predictions in predictions_by_member.values()
            )
        }

        exclusion_reason: str | None = None
        if missing:
            exclusion_reason = (
                f"{len(missing)} member(s) never run: {describe_members(missing)}"
            )
        elif not common_ids:
            exclusion_reason = "no passage is held by every member"

        completeness.append(
            {
                "concept": wikibase_id,
                "gold_passages": len(gold),
                "members_expected": len(expected_members),
                "members_cached": len(predictions_by_member),
                "members_complete": len(predictions_by_member) - len(incomplete),
                "worst_member_passages": min(passages_per_member.values(), default=0),
                "missing_members": describe_members(missing) if missing else None,
                "scored_passages": 0 if exclusion_reason else len(common_ids),
                "excluded": exclusion_reason is not None,
                "exclusion_reason": exclusion_reason,
            }
        )

        if exclusion_reason is not None:
            console.log(
                f"  🚫 Excluding {wikibase_id} from every table: {exclusion_reason}"
            )
            continue

        scored_gold = [passage for passage in gold if passage.id in common_ids]
        if len(scored_gold) < len(gold):
            console.log(
                f"  ✂️  Scoring {len(scored_gold)}/{len(gold)} passages — the rest are "
                "missing from at least one member, so scoring them would put different "
                "ensembles on different passages"
            )
        usable[wikibase_id] = (scored_gold, predictions_by_member)

    return usable, completeness


def build_member_classifier(concept: Concept, member: EnsembleMember) -> LLMClassifier:
    """
    Build the LLM classifier for one ensemble member.

    Uses the default system prompt with no concept-specific labelling guidelines, so the
    only thing describing the concept is what's in the concept store.
    """
    family, seed = member
    return LLMClassifier(
        concept=concept,
        model_name=FAMILIES[family],
        system_prompt_template=LLMClassifierPrompt(
            system_prompt_template=DEFAULT_SYSTEM_PROMPT
        ),
        random_seed=seed,
        temperature=TEMPERATURE,
    )


async def fetch_concept_and_gold(
    wikibase_id: WikibaseID,
) -> tuple[Concept, list[LabelledPassage]] | None:
    """Fetch a concept and its gold-standard passages live from Wikibase and Argilla."""

    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_recursive_has_subconcept=True,
        include_labels_from_subconcepts=True,
    )

    if not concept.labelled_passages:
        console.log(f"⚠️ {wikibase_id} has no labelled passages — skipping")
        return None

    gold = create_gold_standard_labelled_passages(concept.labelled_passages)
    n_positive = sum(1 for passage in gold if passage.spans)
    console.log(
        f"📥 {wikibase_id}: fetched {len(gold)} gold passages, "
        f"{n_positive} positive ({n_positive / len(gold):.1%})"
    )

    return concept, gold


def run_member_with_caching(
    concept: Concept,
    member: EnsembleMember,
    gold: list[LabelledPassage],
    concept_dir: Path,
    batch_size: int,
    position: str,
) -> None:
    """
    Predict one member's passages, reusing whatever it has already cached.

    Only the passages missing from the member's cache are sent to the LLM, so a rerun
    picks up any that a previous run didn't cover — for instance if the gold set grew
    between runs.
    """

    family, seed = member
    cached = read_member_cache(concept_dir, member)
    pending = [passage for passage in gold if passage.id not in cached]

    if not pending:
        console.log(f"{position}: ⏭️  all {len(gold)} passages cached")
        return

    console.log(
        f"{position} (T={TEMPERATURE}): {len(pending)} to predict"
        + (f", {len(cached)} reused from cache" if cached else "")
    )

    try:
        predictions = label_passages_with_classifier(
            classifier=build_member_classifier(concept, member),
            labelled_passages=pending,
            batch_size=batch_size,
            show_progress=True,
        )
    except Exception as e:
        console.log(f"  ⚠️ {family} seed={seed} aborted: {e}. Rerun to resume.")
        return

    # merge with the cache and write back in gold order
    merged = {**cached, **{passage.id: passage for passage in predictions}}
    ordered = [merged[passage.id] for passage in gold if passage.id in merged]
    write_passages(member_path(concept_dir, member), ordered)

    n_positive = sum(1 for passage in ordered if passage.spans)
    console.log(
        f"  ✅ {family} seed={seed}: {len(ordered)}/{len(gold)} passages cached, "
        f"{n_positive} positive ({n_positive / len(ordered):.1%})"
    )
