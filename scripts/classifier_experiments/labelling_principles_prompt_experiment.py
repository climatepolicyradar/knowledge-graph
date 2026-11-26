"""
Compare the hard-coded LLM system prompt with prompts built from Help:Labelling principles.

    # 1. Save the live page's wikitext into the repo (original + a copy to edit)
    uv run python scripts/classifier_experiments/labelling_principles_prompt_experiment.py snapshot
    # 2. Edit labelling_principles/rewritten.wikitext, then run e.g.
    uv run python scripts/classifier_experiments/labelling_principles_prompt_experiment.py run --concept Q618 --option D_page_rewritten
"""

import asyncio
import os
import random
import shutil
from pathlib import Path

import typer
import wandb

from knowledge_graph.classifier.concept_store_prompt import (
    LABELLING_PRINCIPLES_PAGE,
    build_system_prompt_template,
)
from knowledge_graph.classifier.large_language_model import (
    DEFAULT_LLM_CLASSIFIER_PROMPT,
    LLMClassifier,
    LLMClassifierPrompt,
)
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.energy import track_compute_energy_usage
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.evaluate import evaluate_classifier
from knowledge_graph.operations.get_concept import get_concept_async
from knowledge_graph.wikibase import WikibaseSession

PAGE_DIR = Path(__file__).parent / "labelling_principles"
ORIGINAL_WIKITEXT = PAGE_DIR / "original.wikitext"  # don't edit
REWRITTEN_WIKITEXT = PAGE_DIR / "rewritten.wikitext"  # edit this one

# A subset of the concepts used in scripts/benchmarks/eval_set_autolabelling_experiment
# Add additional concepts with --concept.
DEFAULT_CONCEPTS = ["Q618", "Q2029", "Q1277", "Q1837", "Q32"]
# Most eval sets are ~130 passages. Q2029 has 260, so it's capped to match. The same seeded
# sample is used for every option, so options stay paired passage by passage.
DEFAULT_MAX_PASSAGES = 130
SAMPLE_SEED = 42
MODEL_NAME = "openrouter:openai/gpt-5"  # the LLMClassifier default
WANDB_GROUP = "labelling-principles-prompt"
JOB_TYPE = "prompt_experiment"  # keeps these runs apart from train_model runs

# This is wikitext, to append to the bottom of the original page.
APPENDED_LLM_SECTION = """
== Instructions for LLM classifiers ==
These instructions apply when an LLM is labelling passages. Wherever this page refers to the concept's definition in the Concept Store, use the concept description you have been given.

* '''Variant terminology.''' Consider the different ways the concept is referred to in policy documents, including acronyms, abbreviations, technical jargon, and terms that differ between countries and regions.
* '''Indirect mentions.''' Label indirect descriptions of the concept that match its definition, as well as direct references (see ''Principle of comprehensiveness'').
* '''Stricter approaches.''' Where this page says an LLM classifier may need a stricter approach (under ''Principle of liberality'' and ''Lists, tables, headers and fragments''), follow the general rule in that section unless the concept's own labelling guidelines say otherwise.
* '''Final check.''' Before answering, check that you have labelled every instance of the concept that matches its definition, and that each label contains enough of the text to show why it is relevant.
"""

app = typer.Typer()


@app.command()
def snapshot(overwrite: bool = False) -> None:
    """Save the live page's wikitext to original.wikitext, plus a copy to edit as rewritten.wikitext."""
    if ORIGINAL_WIKITEXT.exists() and not overwrite:
        raise typer.BadParameter(
            f"{ORIGINAL_WIKITEXT} exists. Pass --overwrite to replace it."
        )
    wikibase = WikibaseSession()
    PAGE_DIR.mkdir(exist_ok=True)
    ORIGINAL_WIKITEXT.write_text(
        wikibase.get_help_page_wikitext(LABELLING_PRINCIPLES_PAGE)
    )
    if not REWRITTEN_WIKITEXT.exists():
        shutil.copy(ORIGINAL_WIKITEXT, REWRITTEN_WIKITEXT)

    # Rendering the saved wikitext should match the live page exactly
    live = wikibase.get_help_page_text(LABELLING_PRINCIPLES_PAGE)
    rendered = wikibase.render_wikitext_as_text(
        ORIGINAL_WIKITEXT.read_text(), LABELLING_PRINCIPLES_PAGE
    )
    print(
        "✅ Render matches the live page"
        if live == rendered
        else "⚠️ Render differs from the live page"
    )


async def build_options() -> dict[str, LLMClassifierPrompt]:
    """Render every page version through MediaWiki and build each option's prompt."""
    original = ORIGINAL_WIKITEXT.read_text()
    versions = {
        "B_page_as_is": original,
        "C_page_appended": original.rstrip()
        + "\n\n"
        + APPENDED_LLM_SECTION.strip()
        + "\n",
        "D_page_rewritten": REWRITTEN_WIKITEXT.read_text(),
    }
    options = {
        "A_current": DEFAULT_LLM_CLASSIFIER_PROMPT,
        "A2_current_rerun": DEFAULT_LLM_CLASSIFIER_PROMPT,
    }
    async with WikibaseSession() as wikibase:
        for name, wikitext in versions.items():
            text = await wikibase.render_wikitext_as_text_async(
                wikitext, LABELLING_PRINCIPLES_PAGE
            )
            options[name] = LLMClassifierPrompt(
                system_prompt_template=build_system_prompt_template([text])
            )
    return options


@app.command()
def run(
    concept: list[str] = typer.Option(DEFAULT_CONCEPTS),
    option: list[str] | None = typer.Option(
        None, help="Only run these options (default: all)"
    ),
    max_passages: int = typer.Option(
        DEFAULT_MAX_PASSAGES, help="Cap on eval passages per concept"
    ),
) -> None:
    """Evaluate every option on every concept's eval set, one W&B run per concept × option."""
    os.environ["OPENROUTER_API_KEY"] = get_aws_ssm_param(
        "/OpenRouter/KGApiKey", aws_env=AwsEnv.production
    )
    options = asyncio.run(build_options())
    if option:
        options = {name: p for name, p in options.items() if name in option}

    for wikibase_id in concept:
        c = asyncio.run(get_concept_async(wikibase_id=WikibaseID(wikibase_id)))
        passages = c.labelled_passages
        if len(passages) > max_passages:
            passages = random.Random(SAMPLE_SEED).sample(passages, max_passages)
        for option_name, prompt in options.items():
            classifier = LLMClassifier(
                concept=c, model_name=MODEL_NAME, system_prompt_template=prompt
            )
            with wandb.init(
                entity=WANDB_ENTITY,
                project=wikibase_id,
                job_type=JOB_TYPE,
                group=WANDB_GROUP,
                name=f"{option_name}-{wikibase_id}",
                tags=[WANDB_GROUP, option_name],
                config={
                    "option": option_name,
                    "classifier_type": classifier.name,
                    "classifier_id": str(classifier.id),
                    "concept_id": wikibase_id,
                    "model_name": MODEL_NAME,
                    "n_eval_passages": len(passages),
                    "eval_sample_seed": SAMPLE_SEED,
                },
            ) as wandb_run:
                # The exact prompt sent, so every run records which page text it used
                wandb_run.summary["final_system_prompt"] = classifier.system_prompt
                with track_compute_energy_usage() as eval_energy:
                    evaluate_classifier(
                        classifier=classifier,
                        labelled_passages=passages,
                        wandb_run=wandb_run,
                        batch_size=50,
                    )
                eval_energy.log(wandb_run, phase="eval", n_passages=len(passages))


if __name__ == "__main__":
    app()
