"""
Train and sample the root climate justice (Q32) classifiers with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q32_train_llm.py train
    uv run scripts/custom_concept_training/q32_train_llm.py sample
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

WIKIBASE_ID = WikibaseID("Q32")
MODEL_NAME = "openrouter:google/gemini-3-pro-preview"

CONCEPT_DEFINITION = "Justice is the ethical and political framework that addresses fairness, including issues of responsibility, rights, and structural inequity."


def get_concept_description() -> str:
    """
    Build the concept description for justice.

    :returns: The concept description without criteria.
    :rtype: str
    """
    description_q32 = """justice centers the moral obligation to address disproportionate impacts and systemic causes.

In climate, nature and development contexts, this means analysing climate change and the exploitation of natural resources as a political and ethical issue, recognising how climate change -- and the exploitation of natural resources more broadly -- impacts people, communities and countries differently and disproportionately, while benefiting others. Justice requires addressing structural inequities; including those affected and the most vulnerable in decision making; and recognising their well-being, as well as the value of different ways of knowing and being. To be considered justice-aligned, actions must aim to bring about a fairer, more inclusive world."""

    return description_q32


def get_labelling_guidelines() -> str:
    """
    Build the labelling guidelines by fetching related justice concepts from Wikibase.

    :returns: The labelling guidelines with criteria for justice annotation.
    :rtype: str
    """
    console.log("Connecting to Wikibase")
    wikibase = WikibaseSession()

    console.log("Fetching 4 related justice concepts from Wikibase")
    distributive = wikibase.get_concept(WikibaseID("Q911"))
    procedural = wikibase.get_concept(WikibaseID("Q912"))
    # transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria_q32 = f"""
Use the following inclusion/exclusion criteria, in addition to the definition:

THE NORMATIVE REQUIREMENT (Crucial Overall Distinction)
Before tagging, ask: Does this passage make an argument, either explicit
or implicit, that is clearly linked to ethics, fairness or human rights,
for improving the world or the position of vulnerable people?
ONLY tag if the answer is yes.

- **EXCLUDE (Descriptive):** Passages that simply state that inequality, poverty, gender gaps, or labour abuses exist.
  *   *Example (NO):* "Women in this sector often earn less than men" (Describes the problem).
  *   *Example (NO):* "Land tenure insecurity leads to lower agricultural yields for female farmers." (Technical/Impact analysis).
  *   *Example (NO):* "Child labour is common in informal waste picking." (Statement of fact).
- **INCLUDE (Normative/Prescriptive):** Passages that explicitly aim to *address*, *solve*, or *prevent* these issues as a matter of strategy, ethics, or rights.
  *   *Example (YES):* "The policy aims to close the unjust gender pay gap." (Action/Goal).
  *   *Example (YES):* "We must Ensure activities do not exacerbate existing inequalities." (safeguard/remedy).
  *   *Example (YES):* "Strategies to eliminate child labour and uphold human rights." (Solution).
	*   *Example (YES):* "We owe it to our children to ensure they can live fulfilling lives" (ethical argument)

1. PROBLEMS VS. FIXING AND JUDGING 
To be tagged, a passage must not just *describe* an unfair situation; it must *propose* to fix it or *judge* it ethically.
- **INCLUDE**: justice-aligned actions, safeguards, solutions, proposals and actions that prevent further injustice.
- **EXCLUDE**: passages that state an impact on a vulnerable group without providing a link to justice. 

2. ETHICAL ARGUMENTS
- **INCLUDE**: Passages stating why justice is important or why actions should be taken in a just way.

3. JUSTICE-ALIGNED VS. GENERIC ACTION
Avoid tagging buzzwords unless they are specifically qualified by justice concepts (for example, fairness, rights, equity).
  *   *Include:* Meaningful empowerment, especially inclusion/representation in decision making and other actions which meaningfully shift procedural power to marginalised groups.
  *   *Include:* Phrases with a justice-aligned qualifier, such as "ensure **Equal** representation", "**Equitable** access to finance"
  *   *Exclude:* Generic phrases such as "Access to resources." 
  *   *Scrutinize:* Vague and tokenstic language such as "Promoting youth inclusion," "Empowering communities," "Inclusive by design." Tag ONLY if sentence includes information that makes the claim more credible. 
  
4. RIGHTS AND SPECIFIC TARGETS
- **SCRUTINIZE:** Mentions of "The Right to [X]" (e.g., Right to a Healthy Environment) or "Rights of [Group]" (e.g., Rights of Persons with Disabilities). These rights need to be in service of justice or arguing for action on climate, nature, or natural resources.
- **INCLUDE:** Specific SDG-style targets that mention "Equality," "Equity," or eliminating discrimination for named vulnerable groups.
- **INCLUDE:** "Do No Harm" frameworks: Explicit requirements to assess and mitigate risks of widening inequality or violating rights.

5. HOLISTIC JUSTICE
- **INCLUDE**: Passages that describe a specific type of justice. Include in particular procedural justice, distribution, and recognition justice. For this, use the following definitions:
  a) Recognition Justice: {recognition.definition}
  b) Procedural Justice: {procedural.definition}
  c) Distributive Justice: {distributive.definition}
- **INCLUDE**: Ethical discussions of the root causes of inequality and proposals or arguments to fix these root causes.

6.  FORMAT AND CONTEXT EXCLUSIONS
- **SCRUTINIZE:** Ambiguous acronyms (e.g., "JET"). In most cases, you can assume the text comes from a climate- or development context.  INCLUDE if the acronym is likely to be directly and explicitly about justice, but EXCLUDE if justice is only a small component or the context points away from (climate) justice..
- **EXCLUDE:** Fragmented headers or labels (e.g., "&Green Gender Approach") unless they form a complete thought about justice.

7. PRIORITIZE EXPLICIT KEYWORDS
- **INCLUDE:** Passages that explicitly mention justice, including types of justice such as "Intergenerational justice" or "Climate Ethics."
- **SCRUTINIZE:** "Gender," "Poor," "Vulnerable," "Inclusive." Tag these ONLY if they are part of a sentence fulfilling The Normative Requirement.
- **EXCLUDE:** criminal justice and discussions of the justice system, unless they are relevant to climate, development or nature (interpreted broadly). 
"""

    return criteria_q32


@app.command()
def train() -> None:
    """Run training for the justice classifier."""
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
    """Sample passages for the justice concept with custom definition."""
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
