"""
Train and sample distributive justice (Q911) classifiers with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q911_train_llm.py train
    uv run scripts/custom_concept_training/q911_train_llm.py sample
"""

import asyncio

import typer
from rich.console import Console

from knowledge_graph.classifier.large_language_model import (
    DEFAULT_SYSTEM_PROMPT,
    LLMClassifierPrompt,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.sample import main as sample_cli
from scripts.train import run_training

app = typer.Typer()
console = Console()

WIKIBASE_ID = WikibaseID("Q911")
MODEL_NAME = "openrouter:google/gemini-3-pro-preview"

CONCEPT_DEFINITION = "Justice is the ethical and political framework that addresses fairness, including issues of responsibility, rights, and structural inequity."


def get_concept_description() -> str:
    """
    Build the concept description for distributive justice.

    :returns: The concept description without criteria.
    :rtype: str
    """
    description_q911 = """Distributive justice asks 'Who gets what?' based on Equity (need/vulnerability), Responsibility (causal), and Reparation (historical).
In the context of climate change, nature and development, this means ensuring the fair distribution of risks and opportunities of resource exploitation, climate change and the transition to a regenerative economy, cognisant in particular of disproportionate impacts on vulnerable communities and the working class."""

    return description_q911


def get_labelling_guidelines() -> str:
    """
    Build the labelling guidelines by fetching related justice concepts from Wikibase.

    :returns: The labelling guidelines with criteria for distributive justice annotation.
    :rtype: str
    """
    console.log("Connecting to Wikibase")
    wikibase = WikibaseSession()

    console.log("Fetching 2 related justice concepts from Wikibase")
    procedural = wikibase.get_concept(WikibaseID("Q912"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria_q911 = f"""
Use the following inclusion/exclusion criteria, in addition to the definition:

### THE ALLOCATION TEST
Before tagging, ask: Is this passage meaningful for the fair distribution of
resources, costs, or risks based on equity, responsibility, or
historical debt?
ONLY tag if it specifies at least two of the following:
	a) an action (payment, programme, policy, etc) that has redistributional effects
	b) a named marginalized, vulnerable or disproportionally impacted group
	c) an ethical argument or analysis in favour of a fairer distribution of risks, impacts or benefits.

**Examples that FAIL the allocation test:**
- "providing electricity to all citizens" — equality is not necessarily distributive justice; it lacks a targeted equity logic.
- "the project will create 500 jobs" — generic economic benefit without a "Just Transition" framing for specific workers.
- "investing in green technology" — technical investment without an allocation focus.
- "compensating businesses for affected revenue" - absent more context, this could mean small businesses but also big multinationals and their CEOs
- "capacity building for youth" - named group, but no evidence or argument that this will lead to a fairer distribution.

**Examples that PASS the allocation test:**
- "capacity building for youth has led to improved access to finance" - now the benefit for the named group has concrete distributional effects
- "targeted subsidies for low-income households to offset carbon taxes" — addressing disproportionate burdens (Equity).
- "allocation of Loss and Damage funds to Small Island Developing States" — Global North/South spatial justice and historical climate debt.
- "restoration of ancestral lands to communities displaced by coal mining" — Reparative justice/Restoration.
- "ensuring that the costs of climate mitigation do not fall on future generations" — Intergenerational justice.
- "investing in rural cooling centers to protect left-behind agricultural areas" — Spatial justice (Urban/Rural).

### 1. SPATIAL JUSTICE
- **INCLUDE**: Resource allocation that prioritizes specific vulnerable geographies: Global South, rural vs. urban, or "left-behind" regions.
- **INCLUDE**: Moral arguments on the need for fair distributions of harms and benefits between regions or countries.
- **EXCLUDE**: General infrastructure or "aid" that is distributed broadly without a focus on correcting an imbalance.

### 2. INTERGENERATIONAL JUSTICE
- **INCLUDE**: Intergenerational equity—actions taken specifically to protect the rights or resources of future generations.
- **INCLUDE**: Moral arguments on what present generations owe to future generations.
- **EXCLUDE**: Generic inclusion of youth or future generations without any mention of distributive effects.

### 3. HISTORICAL RESPONSIBILITY & REPARATIONS
- **INCLUDE**: "Polluter Pays" mechanisms, climate debt, and reparations/restoration for historical harms.
- **INCLUDE**: Financial transfers from high-emitting countries/entities to those most impacted, as long as some justice-related justification is given
- **EXCLUDE**: Standard commercial insurance or market-rate loans that do not account for historical responsibility.
- **EXCLUDE**: Financial support and project funds where there is no explicit link to justice, fairer distribution or specific targeting of vulnerable groups.

### 4. BURDENS & JUST TRANSITION
- **INCLUDE**: Provisions for workers in declining industries (e.g., "re-skilling for coal miners").
- **INCLUDE**: Measures to prevent "energy poverty" or the "working class" from bearing transition costs.
- **EXCLUDE**: General "poverty reduction" if not linked to climate transition or environmental risks.
- **EXCLUDE**: Generic descriptions of burdens and unfair situations that lack an explicit link to justice or justice-aligned solutions.

### 5. JUSTICE TYPE DIFFERENTIATION
- **EXCLUDE** passages that fit better under:
    - Procedural Justice: {procedural.definition} (Focus on *how* decisions are made).
    - Recognition Justice: {recognition.definition} (Focus on *identity/dignity*).
- **EXCLUDE**: generic mentions of justice and justice policies where there is no explicit distributional element.
"""

    return criteria_q911


@app.command()
def train() -> None:
    """Run training for the distributive justice classifier."""
    concept_overrides_dict = {
        "definition": CONCEPT_DEFINITION,
        "description": get_concept_description(),
    }

    # Build the LLM classifier prompt with labelling guidelines
    labelling_guidelines = get_labelling_guidelines()
    system_prompt_template = LLMClassifierPrompt(
        system_prompt_template=DEFAULT_SYSTEM_PROMPT,
        labelling_guidelines=labelling_guidelines,
    )

    classifier_kwargs = {
        "model_name": MODEL_NAME,
        "system_prompt_template": system_prompt_template,
    }

    console.print("Training model with custom labelling guidelines")
    asyncio.run(
        run_training(
            wikibase_id=WIKIBASE_ID,
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            classifier_type="LLMClassifier",
            classifier_kwargs=classifier_kwargs,
            concept_overrides=concept_overrides_dict,
        )
    )


@app.command()
def sample(
    sample_size: int = typer.Option(10000, help="The number of passages to sample"),
    dataset_name: str = typer.Option(
        "combined",
        help="Dataset to use (balanced or combined)",
    ),
    max_size_to_sample_from: int = typer.Option(
        1000000,
        help="Maximum number of passages to load from the dataset before sampling",
    ),
    track_and_upload: bool = typer.Option(
        True,
        help="Whether to track the run and upload the labelled passages to W&B",
    ),
) -> None:
    """Sample passages for the distributive justice concept with custom definition."""
    concept_overrides = [
        f"definition={CONCEPT_DEFINITION}",
        f"description={get_concept_description()}",
    ]
    sample_cli(
        wikibase_id=WIKIBASE_ID,
        sample_size=sample_size,
        min_negative_proportion=0.1,
        dataset_name=dataset_name,
        max_size_to_sample_from=max_size_to_sample_from,
        track_and_upload=track_and_upload,
        concept_override=concept_overrides,
    )


if __name__ == "__main__":
    app()
