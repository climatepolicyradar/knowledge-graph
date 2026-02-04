"""
Train and sample distributive justice (Q911) classifiers with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q911_train_llm.py train
    uv run scripts/custom_concept_training/q911_train_llm.py sample
"""

import typer
from rich.console import Console

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.sample import main as sample_cli
from scripts.train import main as train_cli

app = typer.Typer()
console = Console()

WIKIBASE_ID = WikibaseID("Q911")
MODEL_NAME = "openrouter:google/gemini-3-pro-preview"

CONCEPT_DEFINITION = "Justice is the ethical and political framework that addresses fairness, including issues of responsibility, rights, and structural inequity."

# I THINK THE BELOW IS NO LONGER USED?
# CHANGED IT ANYWAY TO MAKE SURE IT ALIGNS
INSTRUCTIONS = """
    Instructions:
        1. Read through the passage carefully, thinking about the concept and different ways it is used in documents, including acronyms, jargon and global differences.
        2. Identify any mentions of the concept, including direct references and indirect descriptions of the concept which match the definition. Use the criteria to inform whether the reference meets the threshold.
        3. Surround each identified mention with <concept> tags.
        4. If the passage contains multiple instances, each one should be tagged separately.
        5. If the passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
        6. If the entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
        7. The input text must be reproduced exactly, down to the last character, even if this means typos or other minor formatting issues, only adding concept tags.
        8. Double check that you have tagged all instances of the concept according to the provided definition, and that every tagged part contains enough information to show why this is relevant.
    """


def get_concept_description() -> str:
    """
    Build the concept description by fetching related justice concepts from Wikibase.

    :returns: The full concept description with criteria for procedural justice.
    :rtype: str
    """
    console.log("Connecting to Wikibase")
    wikibase = WikibaseSession()

    console.log("Fetching 3 related justice concepts from Wikibase")
    procedural = wikibase.get_concept(WikibaseID("Q912"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria_q911 = f"""
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
    - **INCLUDE**: Moral arguments on what present generations owe to futuere generations. 
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

    description_q911 = f"""Distributive justice asks 'Who gets what?' based on Equity (need/vulnerability), Responsibility (causal), and Reparation (historical). 
			In the context of climate change, nature and development, this means ensuring the fair distribution of risks and opportunities of resource exploitation, climate change and the transition to a regenerative economy, cognisant in particular of disproportionate impacts on vulnerable communities and the working class. 
			
			In addition to the above definition, use the following criteria to inform your judgement:
			\n{criteria_q911}"""

    return description_q911


def get_concept_overrides() -> list[str]:
    """
    Get the concept override list for training and sampling.

    :returns: List of concept property overrides in key=value format.
    :rtype: list[str]
    """
    concept_description = get_concept_description()
    return [
        f"definition={CONCEPT_DEFINITION}",
        f"description={concept_description}",
    ]


@app.command()
def train() -> None:
    """Run training for the procedural justice classifier."""
    concept_overrides = get_concept_overrides()

    console.print("Training model with default template")
    train_cli(
        wikibase_id=WIKIBASE_ID,
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            f"model_name={MODEL_NAME}",
        ],
        concept_override=concept_overrides,
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
    """Sample passages for the procedural justice concept with custom definition."""
    concept_overrides = get_concept_overrides()
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
