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
    transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria_q32 = f"""
Use the following inclusion/exclusion criteria, in addition to the definition:

### THE ETHICAL TEST
Before tagging, ask: Does this passage make an argument, either explicit
or implicit, that is clearly linked to ethics, fairness or human rights,
for improving the world or the position of vulnerable people?
ONLY tag if the answer is yes.

**Examples that FAIL the ethical test:**
- "climate change is a threat to the economy" — technical/economic framing.
- "we need efficient carbon markets" — utilitarian framing.
- "reducing emissions by 50%" — purely quantitative target.
- "women make up only 1/3 of elected officials" - ethical issue, but no argument for why this should be improved.

**Examples that PASS the ethical test:**
- "Climate change is a threat to human rights so urgent action is warranted."
- "Addressing the structural inequities that drive nature degradation."
- "Climate solutions should ensure that no one is left behind"
- "Making decisions through a fair and inclusive process"
- "A whole-of-society transformation is required"

### 1. STRUCTURAL & HISTORICAL INEQUITY
- **INCLUDE**: Passages linking climate change and nature degradation to colonialism, capitalism, systemic racism, or global power imbalances.
- **INCLUDE**: Discussion of the root causes of vulnerability.
- **NOTE**: Descriptions of unfair situations more generally can ONLY be included if they make an explicit link to root causes, ethical arguments, or describe how/why this needs to be solved.
- **EXCLUDE**: General descriptions of problems, risks or impacts that don't point to an underlying systemic unfairness.

### 2. HOLISTIC JUSTICE
- **INCLUDE**: Passages that describe a specific type of justice. Include in particular procedural, distribution, recognition and transformative justice. For this, use the following definitions:
  a) Recognition Justice: {recognition.definition}
  b) Procedural Justice: {procedural.definition}
  c) Distributive Justice: {distributive.definition}
  d) Transformative Justice: {transformative.definition}
- **INCLUDE**: arguments on "Intergenerational justice" or "Climate Ethics."

### 3. HUMAN RIGHTS & WELL-BEING
- **INCLUDE**: Framings that emphasise universal human rights, including the rights of children, impacted groups, and the specific legal pricniples of the right to a healthy environment and right to development as a matter of justice.
- **EXCLUDE**: Generic human rights mentions that are procedural or tokenistic. These rights need to be clearly in service of justice or arguing for action on climate, nature, or natural resources.

### 4. JUSTICE AS SOLUTION vs. PROBLEM
- **INCLUDE**: Explicit calls for a "Just Transition" or "Justice-aligned actions."
- **EXCLUDE**: Descriptions of barriers (e.g., "corruption is high") unless they also propose a justice-based remedy.

### 5. TOKENISM AND CONTEXT
- **INCLUDE**: Meaningful discussions of- and actions towards a more just world. Draw on your policy knowledge to determine when something is meaningful.
- **INCLUDE**: Names of programmes and policies that are explicitly about justice or fairness.
- **EXCLUDE**: Sentences and phrases where you do not have enough knowledge to determine if they are justice-aligned.
- **EXCLUDE**: Generic policies and programmes, as well as programmes where justice is only a small component.
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
