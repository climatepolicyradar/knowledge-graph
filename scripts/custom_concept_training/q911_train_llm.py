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
MODEL_NAME = "openrouter:google/gemini-3.1-pro-preview"

CONCEPT_DEFINITION = """Distributive justice is the fair allocation of the benefits, burdens, and risks, focusing on those most responsible and those most vulnerable. It includes ethical and political arguments, as well as concrete actions."""


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
Use the following inclusion/exclusion criteria, in addition to the definition.
These are guidelines meant to help you disambiguate. Overall, your main task is to find positive examples. INCLUSION therefore generally trumps EXCLUSION, and an EXCLUSION rule should never rule out whole passages, only specific phrases within that passage.

**Core test:** Distributive justice is the *active* fair allocation of benefits, burdens, and risks. It is not merely the existence of inequality or vulnerability.
You are looking for phrases that contain a mechanism or argument for a fairer distribution, keeping in mind that you are a policy expert who is trained to read between the lines.
 
  **Examples that FAIL the test:**
    - "providing electricity to all citizens" — equality is not necessarily distributive justice; this cannot be included unless it mentions how this lifts up historically underserved or vulnerable groups.
    - "the project will create 500 jobs" — generic economic benefit without a "Just Transition" framing for specific workers.  
	- "compensating businesses for affected revenue" - absent more context, this could mean small businesses but also big multinationals and their CEOs
	- "capacity building for youth" - named group, but no evidence or argument that this will lead to a fairer distribution.

    **Examples that PASS the test:**
    - "capacity building for youth has led to improved access to finance" - now the benefit for the named group has concrete distributional effects
    - "allocation of Loss and Damage funds to Small Island Developing States" — Global North/South spatial justice and historical climate debt.
    - "restoration of ancestral lands to communities displaced by coal mining" — Reparative justice/Restoration.
    - "fairness demands that the costs of climate mitigation do not fall on future generations" — ethical argument for intergenerational justice.

### 1. THE ACTION/ARGUMENT REQUIREMENT
To be tagged, the passage must do more than mention a vulnerable group or a "Just Transition" keyword. It must contain a **proposition**—either a concrete policy action or a normative argument.
*   **Action:** INCLUDE if the text describes a transfer of resources, rights, or opportunities *specifically to* a vulnerable group or region, and/or redistribution *away from* a party that has caused harm.
*   **Argument:** INCLUDE if the text claims that such a distribution *ought* to happen based on fairness, equity, or historical responsibility.

### 2. INCLUSION CRITERIA
Tag the passage if it matches any of the following:

**A. Active Redistribution & Targeting**
*   **INCLUDE** specific measures to support groups disproportionately impacted by climate change or transition, interpreted broadly (e.g., "providing grants to coal communities," "social protection for women farmers").
*   **INCLUDE** allocations of funding or technology that explicitly prioritize "left-behind" regions or marginalized demographics, again interpreted broadly (e.g. Global South, rural poor) to correct an imbalance.
*   **INCLUDE**: social security and pension measures which target, broadly, vulnerable areas or groups, or which otherwise mention a fairer distribution of benefits or harms
*   **INCLUDE**: specific financial, infrastructural, or legal mechanisms directed explicitly at vulnerable or marginalized groups (e.g. 'debt redemption funds for farmers', 'green financing for smallholders', 'targeted land titling for women'). The distributive intent is inherent in the targeting of the mechanism."
*   **INCLUDE**: generic mechanisms that inherently involve redistribution or targeted benefits, such as 'benefit-sharing mechanisms' or strategies specifically designed for 'impacted workers and communities'.

**B. Burden Sharing & Compensation**
*   **INCLUDE** mechanisms to mitigate the negative social impacts of green transition (e.g., "compensation for lost livelihoods," "retraining funds for miners").
*   **INCLUDE** arguments for- or policies enforcing "Polluter Pays" (making high-emitters bear the costs) or "Common But Differentiated Responsibilities" (fair sharing of global climate effort).

**C. Ethical imperatives**
*   **INCLUDE** statements dealing with *requirements* or *imperatives* for justice (e.g., "Significant concessionary funds will be *required* for the region to participate," "Programmes *must* target vulnerable groups").
*   **INCLUDE** arguments for a fairer distribution of support, resources or burdens that appeal to ethics or justice. The ethical argument here can be descriptive, as long as the distributive component is clearly present.

**D. Distributive justice sub-types**
*   **INCLUDE** more specific versions of distributive justice, including:
	- Intergenerational justice (fair distribution between present and future generations)
	- Spatial justice (fair distribution between areas and countries)
	- Restorative justice (compensation, reparations or systemic change from parties that historically caused harm towards parties that have been harmed).   

### 3. EXCLUSION CRITERIA
Do **NOT** tag the passage if it falls into these categories, even if it contains keywords like "women," "vulnerable," "inequality," or "transition":
Be strict here: we are looking for substantive mentions, not greenwashing measures.

**A. Describing the State of Affairs**
*   **Exclude** factual descriptions or statistics regarding poverty, gender gaps, or lack of resources (e.g., "Women earn 20% less than men," "Rural areas lack access to water").
*   **Exclude** descriptions of vulnerability that do not mention remedial action (e.g. "The elderly are most vulnerable to heatwaves").
    *   *Reasoning:* Describing a problem is not the same as proposing a distributive justice solution. If the passage stays purely at the problem level, exclude. Of course, if this problem description is preceded/followed by a targeted (argument for a) solution, the description is useful and the whole passage, including the solution, should be included.

**B. Describing current or future impacts**
*   **Exclude** predictions of economic decline, job losses, or "ghost towns" resulting from the transition *unless* the passage also discusses the measures to address them.
    *   *Example:* "The coal industry will lose 10,000 jobs" -> **EXCLUDE**.
    *   *Example:* "We are launching a fund to support the 10,000 workers losing jobs" -> **INCLUDE**.

**C. Nominal Mentions & Titles**
*   **Exclude** titles, headers, proper nouns, or names of committees/documents (e.g. "Just Transition Mechanism," "Section 7: Financing").
*   **Exclude** citations or references where the term is just a label (e.g., "As discussed in the 2021 Just Transition Report...").
*   **NOTE**: Project/policy titles are often too generic to be included, but can be included if they explicitly define a targeted resource transfer to a vulnerable group (e.g., 'Inclusive Green Financing for Smallholder Agriculture')."

**D. Procedural & Technical Details**
*   **Exclude** purely administrative text, such as eligibility checklists, reporting deadlines, or definitions of terms (e.g., "Beneficiaries must show ID," "Polluter is defined as...").
*   **NOTE** generic terms such as "capacity building" or "technical assistance" are generally to be excluded, unless there is an explicit statement explaining how this redistributes power or resources to a marginalized group to correct an injustice.

**E. Generic Social Welfare**
*   **Exclude** social security or pension provisions that are standard functions of a state, UNLESS they are explicitly framed as addressing a specific vulnerability, historical bias, or transition impact.

**F. Other/generic justice types
*   **Exclude** passages that fit better under:
        - Procedural Justice: {procedural.definition}
        - Recognition Justice: {recognition.definition}
*   **Exclude**: generic mentions of justice and justice policies. For a passage to be Distributive Justice, there must be an allocation of tangible benefits, resources, funding, or compensation.
   

### 4. AMBIGUITY CHECK
*   **If the text implies "fairness" but is vague:** Ask, "Does this text explain *who* gets *what* and *why*?" If it provides an answer to at least two elements, **INCLUDE**.
*   **If the text is a goal statement:** "Objective: Allocate 40% of project funding to female-headed households." -> **INCLUDE** (This sets a distributive aim).
*   **If the text does not match any of the criteria, go back to the definition and the core test.**
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
