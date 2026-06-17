"""
Self-improving keyword expansion, evaluated against the concept store.

The concept store carries human-curated ``alternative_labels`` for each concept.
We treat those as ground truth: hide them, ask the keyword-expander LLM to
regenerate them from the concept's other information, and score the overlap
(exact set match → precision/recall/F-beta). A "teacher"/optimiser LLM then reads
those results and rewrites the expander's prompt to maximise the score — the same
idea as :mod:`knowledge_graph.classifier.autollm`, but applied to keyword
generation rather than passage classification.

To avoid the teacher overfitting to the concepts we report on, evaluation uses
k-fold cross-validation: in each fold the teacher only ever sees results from the
train split, and the held-out fold is scored solely by the selected template.
"""

import asyncio
import json
import random
import re
import statistics
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import cast

import pandas as pd
from mypy_boto3_s3.client import S3Client
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.classifier.keyword_expansion import (
    DEFAULT_KEYWORD_EXPANSION_PROMPT,
    KeywordExpansionClassifier,
)
from knowledge_graph.cloud import AwsEnv, get_s3_client
from knowledge_graph.concept import Concept
from knowledge_graph.config import processed_data_dir
from knowledge_graph.metrics import ConfusionMatrix

# The balanced passage sample produced by flows/build_dataset_flow.py.
SAMPLED_DATASET_S3_BUCKET = "cpr-kg-feather-files"
SAMPLED_DATASET_S3_KEY = "build_dataset/sampled_dataset.feather"
SAMPLED_DATASET_REGION = "eu-west-1"
PASSAGE_TEXT_COLUMN = "text_block.text"


def load_sampled_passages(
    limit: int | None = None,
    seed: int = 42,
    aws_env: AwsEnv = AwsEnv.production,
) -> list[str]:
    """
    Load the balanced sampled passage dataset as a list of text strings.

    Reads the local cache (``data/processed/sampled_dataset.feather``) if present,
    otherwise downloads it from S3 and caches it there.

    :param limit: Optionally sample down to this many passages (seeded)
    :param seed: Seed for the sample
    :param aws_env: AWS environment to use when downloading from S3
    :return: List of passage texts
    """
    local_path = processed_data_dir / "sampled_dataset.feather"
    if local_path.exists():
        df = pd.read_feather(local_path)
    else:
        print(
            f"Downloading sampled dataset from "
            f"s3://{SAMPLED_DATASET_S3_BUCKET}/{SAMPLED_DATASET_S3_KEY}"
        )
        s3_client = cast(S3Client, get_s3_client(aws_env, SAMPLED_DATASET_REGION))
        response = s3_client.get_object(
            Bucket=SAMPLED_DATASET_S3_BUCKET, Key=SAMPLED_DATASET_S3_KEY
        )
        body = response["Body"].read()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(body)
        df = pd.read_feather(BytesIO(body))

    texts = [t for t in df[PASSAGE_TEXT_COLUMN].dropna().tolist() if str(t).strip()]
    if limit is not None and len(texts) > limit:
        texts = random.Random(seed).sample(texts, limit)
    print(f"Loaded {len(texts)} sampled passages")
    return texts


DEFAULT_META_PROMPT = """
You are an expert prompt engineer improving a keyword-generation system.

A "keyword expander" LLM is given a concept (its name, and sometimes a description
and definition) and must output keywords. A simple string-matching classifier then
uses those keywords to find mentions of the concept in policy documents. We measure
quality FUNCTIONALLY: we build a classifier from the generated keywords and one from
the human-curated keywords, run both over a sample of real passages, and measure how
well the generated classifier reproduces the passages the human classifier matches
(precision, recall, F-beta). So generating a different-but-equivalent synonym is fine
if it matches the same passages; generating terms that match unrelated passages hurts
precision, and failing to match passages the human keywords catch hurts recall.

Here is the CURRENT prompt template given to the expander. It is a Python format
string with two placeholders, {{PREFERRED_LABEL}} and {{CONCEPT_DESCRIPTION}}, and it
instructs the model to return JSON. Note that literal JSON braces are escaped as
double braces ({{{{ and }}}}) so that Python's str.format works:

<current_prompt_template>
{current_prompt_template}
</current_prompt_template>

Below are evaluation results across several concepts. For each you can see what the
expander generated, the human gold keywords, and the precision/recall achieved:

<evaluation_results>
{evaluation_results}
</evaluation_results>

Analyse where the expander is going wrong:
1. Low recall — what kinds of valid keywords is it failing to generate? (missed
   synonyms, acronyms, morphological variants, alternative spellings...)
2. Low precision — what kinds of spurious keywords is it generating that humans would
   not include? (overly broad terms, loosely related concepts, near-duplicates...)
3. Look for patterns that GENERALISE across concepts, not facts about individual
   concepts.

Then rewrite the prompt template to fix these issues.

Hard requirements for `new_prompt_template`:
- It MUST contain both {{PREFERRED_LABEL}} and {{CONCEPT_DESCRIPTION}} exactly once each.
- It MUST instruct the model to return valid JSON with "positive_keywords" and
  "negative_keywords" lists. Any literal JSON braces in the template MUST be escaped
  as double braces ({{{{ }}}}) so that Python's str.format works.
- Improve the GUIDANCE, not the placeholders. Do NOT include any concept-specific
  keywords or examples drawn from the evaluation results above — that would overfit to
  the evaluation set. If you add illustrative examples, invent neutral generic ones.
"""

# Keep separator handling consistent with KeywordClassifier so that e.g.
# "greenhouse gas", "greenhouse-gas" and "greenhouse–gas" normalise to one form.
_SEPARATOR_RE = re.compile(r"[\s\-–—]+")


def normalise_keyword(keyword: str) -> str:
    """Lowercase, strip, and collapse separator characters to a single space."""
    return _SEPARATOR_RE.sub(" ", keyword.strip().lower()).strip()


def score_keyword_sets(generated: list[str], gold: list[str]) -> ConfusionMatrix:
    """
    Score generated keywords against a gold set as exact (normalised) set overlap.

    :param list[str] generated: Keywords produced by the expander
    :param list[str] gold: Human-curated ground-truth keywords
    :return ConfusionMatrix: TP = overlap, FP = generated-only, FN = gold-only
    """
    generated_set = {n for k in generated if (n := normalise_keyword(k))}
    gold_set = {n for k in gold if (n := normalise_keyword(k))}

    return ConfusionMatrix(
        true_positives=len(generated_set & gold_set),
        false_positives=len(generated_set - gold_set),
        false_negatives=len(gold_set - generated_set),
    )


def _clean_keyword_lists(
    concept: Concept, positives: list[str], negatives: list[str]
) -> tuple[list[str], list[str]]:
    """
    Clean keyword lists so a Concept copy can be built without validation issues.

    Drops the preferred label and duplicates from positives, and drops negatives that
    are empty or overlap the positives.
    """
    preferred = concept.preferred_label.strip()
    seen_pos: set[str] = set()
    pos: list[str] = []
    for keyword in positives:
        cleaned = keyword.strip()
        if not cleaned or cleaned == preferred:
            continue
        if cleaned.lower() not in seen_pos:
            seen_pos.add(cleaned.lower())
            pos.append(cleaned)

    positive_lower = seen_pos | {preferred.lower()}
    seen_neg: set[str] = set()
    neg: list[str] = []
    for keyword in negatives:
        cleaned = keyword.strip()
        if not cleaned or cleaned.lower() in positive_lower:
            continue
        if cleaned.lower() not in seen_neg:
            seen_neg.add(cleaned.lower())
            neg.append(cleaned)

    return pos, neg


def _keyword_classifier(
    concept: Concept, positives: list[str], negatives: list[str]
) -> KeywordClassifier:
    """Build a KeywordClassifier matching a given set of positive/negative keywords."""
    pos, neg = _clean_keyword_lists(concept, positives, negatives)
    keyword_concept = concept.model_copy(
        update={"alternative_labels": sorted(pos), "negative_labels": sorted(neg)}
    )
    return KeywordClassifier(keyword_concept)


def _matched_passage_indices(
    classifier: KeywordClassifier, passages: list[str], batch_size: int
) -> set[int]:
    """Return the indices of passages in which the classifier finds >=1 span."""
    predictions = classifier.predict(passages, batch_size=batch_size)
    return {i for i, spans in enumerate(predictions) if spans}


def functional_confusion(
    generated_indices: set[int], human_indices: set[int], n_passages: int
) -> ConfusionMatrix:
    """
    Score the generated classifier's matches against the human classifier's matches.

    The human classifier (built from the concept's curated labels) is the reference.

    :param generated_indices: Passage indices matched by the generated-keyword classifier
    :param human_indices: Passage indices matched by the human-keyword classifier
    :param n_passages: Total number of passages evaluated
    :return ConfusionMatrix: agreement of generated matches vs human matches
    """
    return ConfusionMatrix(
        true_positives=len(generated_indices & human_indices),
        false_positives=len(generated_indices - human_indices),
        false_negatives=len(human_indices - generated_indices),
        true_negatives=n_passages - len(generated_indices | human_indices),
    )


class EvalMode(str, Enum):
    """How much of the concept the expander is allowed to see during evaluation."""

    LABELS_HIDDEN = "labels_hidden"  # preferred label + description + definition
    MINIMAL = "minimal"  # preferred label only


def masked_markdown(concept: Concept, mode: EvalMode) -> str:
    """
    Render a concept to markdown with gold labels removed.

    In MINIMAL mode the description/definition are also removed, so the expander
    cannot simply copy them.
    """
    return concept.to_markdown(
        include_description=mode is EvalMode.LABELS_HIDDEN,
        include_definition=mode is EvalMode.LABELS_HIDDEN,
        include_alternative_labels=False,
        include_negative_labels=False,
        include_concept_neighbourhood=False,
        include_example_passages=False,
    )


def _template_is_formattable(template: str) -> bool:
    """Whether ``template`` keeps both placeholders and survives str.format."""
    if "{PREFERRED_LABEL}" not in template or "{CONCEPT_DESCRIPTION}" not in template:
        return False
    try:
        template.format(PREFERRED_LABEL="x", CONCEPT_DESCRIPTION="y")
    except (KeyError, IndexError, ValueError):
        return False
    return True


@dataclass
class KeywordEvalResult:
    """The result of evaluating one concept under one mode."""

    concept_id: str
    preferred_label: str
    mode: EvalMode
    generated_positives: list[str]
    gold_positives: list[str]
    # Primary metric: functional agreement of matched passages vs the human classifier.
    confusion_matrix: ConfusionMatrix
    # Secondary readout: exact (normalised) string overlap with the gold labels.
    string_confusion_matrix: ConfusionMatrix


@dataclass
class TrialResult:
    """The result of evaluating one prompt template across a set of concepts."""

    prompt_template: str
    f_beta_score: float
    results: list[KeywordEvalResult]


class OptimiserResponse(BaseModel):
    """Response from the teacher model containing a rewritten expansion prompt."""

    analysis: str = Field(
        description="Brief analysis of the results, identifying generalisable patterns "
        "in missed and spurious keywords"
    )
    new_prompt_template: str = Field(
        description="Improved keyword expansion prompt template, keeping the "
        "{PREFERRED_LABEL} and {CONCEPT_DESCRIPTION} placeholders and the JSON output "
        "format (with literal braces escaped as double braces)"
    )

    @field_validator("new_prompt_template")
    @classmethod
    def _check_placeholders(cls, value: str) -> str:
        for placeholder in ("{PREFERRED_LABEL}", "{CONCEPT_DESCRIPTION}"):
            if placeholder not in value:
                raise ValueError(
                    f"new_prompt_template must contain the {placeholder} placeholder"
                )
        return value


def _aggregate(results: list[KeywordEvalResult]) -> ConfusionMatrix:
    """Micro-average a list of per-concept results into one confusion matrix."""
    return ConfusionMatrix(
        true_positives=sum(r.confusion_matrix.true_positives for r in results),
        false_positives=sum(r.confusion_matrix.false_positives for r in results),
        false_negatives=sum(r.confusion_matrix.false_negatives for r in results),
    )


def _append_jsonl(path: Path | None, row: dict) -> None:
    """Append a single result row to a JSONL file (no-op if path is None)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _format_results_for_teacher(
    results: list[KeywordEvalResult], max_examples: int = 8
) -> str:
    """Render evaluation results into a compact block for the teacher prompt."""
    blocks = []
    for r in results:
        generated = {normalise_keyword(k) for k in r.generated_positives}
        gold = {normalise_keyword(k) for k in r.gold_positives}
        missed = sorted(gold - generated)[:max_examples]
        extra = sorted(generated - gold)[:max_examples]
        blocks.append(
            f"Concept: {r.preferred_label} (mode={r.mode.value})\n"
            f"  functional match vs human keywords: "
            f"precision={r.confusion_matrix.precision():.2f} "
            f"recall={r.confusion_matrix.recall():.2f}\n"
            f"  human keywords the generated set did NOT reproduce: {missed}\n"
            f"  generated keywords not in the human set "
            f"(some valid synonyms, some spurious): {extra}"
        )
    return "\n\n".join(blocks)


def _evaluate_template(
    template: str,
    concepts: list[Concept],
    modes: tuple[EvalMode, ...],
    expander_model: str,
    beta: float,
    passages: list[str],
    human_matches: dict[str, set[int]],
    batch_size: int,
    results_path: Path | None,
    fold: int,
    split: str,
    trial: int,
) -> TrialResult:
    """
    Evaluate one prompt template across concepts × modes, writing rows incrementally.

    The primary score is FUNCTIONAL: a classifier built from the generated keywords is
    run over ``passages`` and its matches are compared against ``human_matches`` (the
    passages matched by the concept's curated keywords). A secondary exact string
    overlap is also recorded.

    A single expander instance is reused across all concepts because keyword
    generation depends only on the template, not on the concept it was constructed
    with.
    """
    expander = KeywordExpansionClassifier(
        concept=concepts[0], model=expander_model, prompt_template=template
    )
    n_passages = len(passages)

    results: list[KeywordEvalResult] = []
    for concept in concepts:
        for mode in modes:
            generated = expander._generate_keywords(
                preferred_label=concept.preferred_label,
                concept_description=masked_markdown(concept, mode),
            )

            generated_classifier = _keyword_classifier(
                concept,
                generated["positive_keywords"],
                generated["negative_keywords"],
            )
            generated_indices = _matched_passage_indices(
                generated_classifier, passages, batch_size
            )
            human_indices = human_matches[str(concept.wikibase_id)]
            functional_cm = functional_confusion(
                generated_indices, human_indices, n_passages
            )
            string_cm = score_keyword_sets(
                generated["positive_keywords"], concept.alternative_labels
            )

            result = KeywordEvalResult(
                concept_id=str(concept.wikibase_id),
                preferred_label=concept.preferred_label,
                mode=mode,
                generated_positives=generated["positive_keywords"],
                gold_positives=concept.alternative_labels,
                confusion_matrix=functional_cm,
                string_confusion_matrix=string_cm,
            )
            results.append(result)
            _append_jsonl(
                results_path,
                {
                    "fold": fold,
                    "split": split,
                    "trial": trial,
                    "concept_id": result.concept_id,
                    "preferred_label": result.preferred_label,
                    "mode": mode.value,
                    "generated_positives": generated["positive_keywords"],
                    "generated_negatives": generated["negative_keywords"],
                    "gold_positives": concept.alternative_labels,
                    "gold_negatives": concept.negative_labels,
                    "n_passages": n_passages,
                    "human_matched_passages": len(human_indices),
                    "generated_matched_passages": len(generated_indices),
                    "functional_true_positives": functional_cm.true_positives,
                    "functional_false_positives": functional_cm.false_positives,
                    "functional_false_negatives": functional_cm.false_negatives,
                    "functional_precision": functional_cm.precision(),
                    "functional_recall": functional_cm.recall(),
                    "functional_f_beta": functional_cm.f_beta_score(beta=beta),
                    "string_precision": string_cm.precision(),
                    "string_recall": string_cm.recall(),
                    "string_f_beta": string_cm.f_beta_score(beta=beta),
                },
            )

    f_beta = _aggregate(results).f_beta_score(beta=beta)
    return TrialResult(prompt_template=template, f_beta_score=f_beta, results=results)


def _optimise_on(
    train_concepts: list[Concept],
    modes: tuple[EvalMode, ...],
    optimiser_agent: Agent[None, OptimiserResponse],
    n_trials: int,
    beta: float,
    expander_model: str,
    passages: list[str],
    human_matches: dict[str, set[int]],
    batch_size: int,
    results_path: Path | None,
    fold: int,
) -> TrialResult:
    """
    Iteratively improve the expansion prompt on a single train split.

    The teacher only ever sees results from ``train_concepts``. Returns the
    best-on-train :class:`TrialResult`.
    """
    print(
        f"  Trial 0: evaluating the default expansion prompt on {len(train_concepts)} concepts"
    )
    best = _evaluate_template(
        template=DEFAULT_KEYWORD_EXPANSION_PROMPT,
        concepts=train_concepts,
        modes=modes,
        expander_model=expander_model,
        beta=beta,
        passages=passages,
        human_matches=human_matches,
        batch_size=batch_size,
        results_path=results_path,
        fold=fold,
        split="train",
        trial=0,
    )
    print(f"  Trial 0 train F-beta: {best.f_beta_score:.4f}")

    current = best
    for trial in range(1, n_trials + 1):
        meta_prompt = DEFAULT_META_PROMPT.format(
            current_prompt_template=current.prompt_template,
            evaluation_results=_format_results_for_teacher(current.results),
        )
        response = optimiser_agent.run_sync(meta_prompt).output
        print(f"  Trial {trial} teacher analysis:\n{response.analysis}")

        candidate = response.new_prompt_template
        if not _template_is_formattable(candidate):
            print(
                f"  Trial {trial}: proposed template failed validation; "
                "keeping current template"
            )
            candidate = current.prompt_template

        trial_result = _evaluate_template(
            template=candidate,
            concepts=train_concepts,
            modes=modes,
            expander_model=expander_model,
            beta=beta,
            passages=passages,
            human_matches=human_matches,
            batch_size=batch_size,
            results_path=results_path,
            fold=fold,
            split="train",
            trial=trial,
        )
        print(f"  Trial {trial} train F-beta: {trial_result.f_beta_score:.4f}")

        current = trial_result
        if trial_result.f_beta_score > best.f_beta_score:
            best = trial_result

    print(f"  Best train F-beta on fold {fold}: {best.f_beta_score:.4f}")
    return best


def _make_folds(concepts: list[Concept], k: int, seed: int) -> list[list[Concept]]:
    """Shuffle (seeded) and partition concepts into k disjoint folds."""
    shuffled = list(concepts)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def _metrics(results: list[KeywordEvalResult], beta: float) -> dict[str, float]:
    """Micro-averaged precision, recall, F-beta and keyword count over results."""
    cm = _aggregate(results)
    return {
        "precision": cm.precision(),
        "recall": cm.recall(),
        "f_beta": cm.f_beta_score(beta=beta),
        # Number of distinct gold + generated keywords involved (tp + fp + fn).
        "n_keywords": cm.true_positives + cm.false_positives + cm.false_negatives,
    }


def _per_mode_breakdown(
    results: list[KeywordEvalResult], modes: tuple[EvalMode, ...], beta: float
) -> dict[str, dict[str, float]]:
    """Precision/recall/F-beta for each mode over the given results."""
    return {
        mode.value: _metrics([r for r in results if r.mode is mode], beta)
        for mode in modes
    }


def cross_validate(
    concepts: list[Concept],
    passages: list[str],
    k: int = 5,
    modes: tuple[EvalMode, ...] = (EvalMode.LABELS_HIDDEN, EvalMode.MINIMAL),
    n_trials: int = 3,
    beta: float = 1.0,
    optimiser_model_name: str = "openrouter:google/gemini-3.1-pro-preview",
    expander_model: str = "openrouter:google/gemini-3-flash-preview",
    batch_size: int = 200,
    results_path: Path | None = Path(
        "data/processed/keyword_expansion_optimisation.jsonl"
    ),
    seed: int = 42,
    produce_final_template: bool = True,
) -> dict:
    """
    Estimate the keyword-expansion-optimisation procedure via k-fold cross-validation.

    In each fold the teacher optimises the prompt on the other k-1 folds and the
    held-out fold is scored only by the selected template, giving an unbiased
    estimate of the procedure's quality. Scoring is FUNCTIONAL: generated keywords are
    turned into a classifier and run over ``passages``, then compared against the
    passages matched by each concept's curated keywords.

    :param list[Concept] concepts: Eval concepts (should each have >=2 alternative labels)
    :param list[str] passages: Real passages to run the keyword classifiers over
    :param int k: Number of cross-validation folds
    :param tuple[EvalMode, ...] modes: Evaluation modes; each concept is scored under all
    :param int n_trials: Optimisation trials per fold (excluding the trial-0 baseline)
    :param float beta: Beta for the F-beta score (1.0 = F1; must be in [0, 1])
    :param str optimiser_model_name: pydantic-ai model name for the teacher
    :param str expander_model: model name for the keyword expander
    :param int batch_size: Batch size for running keyword classifiers over passages
    :param Path | None results_path: Where to append per-result JSONL rows (None = no writes)
    :param int seed: Seed for the fold shuffle
    :param bool produce_final_template: Whether to also optimise over all concepts to
        produce a deliverable prompt template
    :return dict: ``{"cv_mean", "cv_std", "per_fold", "deliverable_template", ...}``
    """
    if k < 2:
        raise ValueError("k must be at least 2 for cross-validation")
    if len(concepts) < k:
        raise ValueError(
            f"Need at least k={k} concepts to cross-validate, got {len(concepts)}"
        )

    # Precompute, once, the passages each concept's curated keywords match (reference).
    print(f"Computing human-keyword matches over {len(passages)} passages...")
    human_matches = {
        str(c.wikibase_id): _matched_passage_indices(
            KeywordClassifier(c), passages, batch_size
        )
        for c in concepts
    }

    folds = _make_folds(concepts, k, seed)

    n_modes = len(modes)
    avg_train = len(concepts) * (k - 1) / k
    projected_generations = int(
        k * (n_trials + 1) * avg_train * n_modes  # train evaluations
        + k * (len(concepts) / k) * n_modes  # held-out evaluations
    )
    print(
        f"Cross-validating over {len(concepts)} concepts, k={k}, n_trials={n_trials}, "
        f"modes={[m.value for m in modes]}\n"
        f"Projected expander generation calls: ~{projected_generations} "
        f"(+ {k * n_trials} teacher calls)"
    )

    optimiser_agent = Agent(model=optimiser_model_name, output_type=OptimiserResponse)

    per_fold = []
    all_held_out: list[KeywordEvalResult] = []
    for fold_idx, test_concepts in enumerate(folds):
        train_concepts = [c for f in folds if f is not test_concepts for c in f]
        print("=" * 60)
        print(
            f"Fold {fold_idx + 1}/{k}: train={len(train_concepts)} "
            f"held-out={len(test_concepts)}"
        )
        print("=" * 60)

        best = _optimise_on(
            train_concepts=train_concepts,
            modes=modes,
            optimiser_agent=optimiser_agent,
            n_trials=n_trials,
            beta=beta,
            expander_model=expander_model,
            passages=passages,
            human_matches=human_matches,
            batch_size=batch_size,
            results_path=results_path,
            fold=fold_idx,
        )

        held_out = _evaluate_template(
            template=best.prompt_template,
            concepts=test_concepts,
            modes=modes,
            expander_model=expander_model,
            beta=beta,
            passages=passages,
            human_matches=human_matches,
            batch_size=batch_size,
            results_path=results_path,
            fold=fold_idx,
            split="test",
            trial=-1,
        )
        all_held_out.extend(held_out.results)
        fold_metrics = _metrics(held_out.results, beta)
        print(
            f"Fold {fold_idx + 1} held-out: "
            f"precision={fold_metrics['precision']:.4f} "
            f"recall={fold_metrics['recall']:.4f} "
            f"F-beta={fold_metrics['f_beta']:.4f}"
        )
        per_fold.append(
            {
                "fold": fold_idx,
                "held_out": fold_metrics,
                "held_out_per_mode": _per_mode_breakdown(held_out.results, modes, beta),
                "best_train_f_beta": best.f_beta_score,
            }
        )

    held_out_scores = [f["held_out"]["f_beta"] for f in per_fold]
    cv_mean = statistics.mean(held_out_scores)
    cv_std = statistics.pstdev(held_out_scores)
    # Pool every held-out result for a single micro-averaged precision/recall.
    pooled = _metrics(all_held_out, beta)
    pooled_per_mode = _per_mode_breakdown(all_held_out, modes, beta)
    print("=" * 60)
    print(
        f"Cross-validated held-out F-beta: {cv_mean:.4f} ± {cv_std:.4f} (mean over folds)"
    )
    print(
        f"Pooled held-out: precision={pooled['precision']:.4f} "
        f"recall={pooled['recall']:.4f} F-beta={pooled['f_beta']:.4f}"
    )
    for mode_name, m in pooled_per_mode.items():
        print(
            f"  [{mode_name}] precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} F-beta={m['f_beta']:.4f}"
        )
    print("=" * 60)

    deliverable_template = None
    if produce_final_template:
        print("Producing deliverable template by optimising over all concepts...")
        final = _optimise_on(
            train_concepts=concepts,
            modes=modes,
            optimiser_agent=optimiser_agent,
            n_trials=n_trials,
            beta=beta,
            expander_model=expander_model,
            passages=passages,
            human_matches=human_matches,
            batch_size=batch_size,
            results_path=results_path,
            fold=-1,
        )
        deliverable_template = final.prompt_template
        print("Deliverable prompt template:\n")
        print(deliverable_template)

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "pooled_held_out": pooled,
        "pooled_held_out_per_mode": pooled_per_mode,
        "per_fold": per_fold,
        "deliverable_template": deliverable_template,
    }


async def _load_eval_concepts(
    max_concepts: int, min_alternative_labels: int, seed: int, limit: int | None
) -> list[Concept]:
    """Fetch concepts from the store and sample those with enough alternative labels."""
    from knowledge_graph.wikibase import WikibaseSession

    async with WikibaseSession() as wikibase:
        print("Fetching concepts from Wikibase...")
        concepts = await wikibase.get_concepts_async(limit=limit)

    eligible = [
        c for c in concepts if len(c.alternative_labels) >= min_alternative_labels
    ]
    print(
        f"{len(eligible)}/{len(concepts)} concepts have "
        f">= {min_alternative_labels} alternative labels"
    )

    rng = random.Random(seed)
    if len(eligible) > max_concepts:
        eligible = rng.sample(eligible, max_concepts)
    return eligible


if __name__ == "__main__":
    eval_concepts = asyncio.run(
        _load_eval_concepts(
            max_concepts=25,
            min_alternative_labels=2,
            seed=42,
            limit=None,
        )
    )

    eval_passages = load_sampled_passages(limit=5000)

    cross_validate(
        eval_concepts,
        passages=eval_passages,
        k=5,
        n_trials=3,
        beta=1.0,
    )
