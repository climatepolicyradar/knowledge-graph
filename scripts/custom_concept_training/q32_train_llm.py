"""
Train and sample the root climate justice (Q32) classifiers with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q32_train_llm.py train
    uv run scripts/custom_concept_training/q32_train_llm.py sample
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

WIKIBASE_ID = WikibaseID("Q32")
MODEL_NAME = "openrouter:google/gemini-3-pro-preview"

CONCEPT_DEFINITION = "Justice is the ethical and political framework that addresses fairness, including issues of responsibility, rights, and structural inequity."

#I THINK THE BELOW IS NO LONGER USED?
#CHANGED IT ANYWAY TO MAKE SURE IT ALIGNS
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
    distributive = wikibase.get_concept(WikibaseID("Q911"))
    procedural = wikibase.get_concept(WikibaseID("Q912"))
    transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria_q32 = f"""
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

    description_q32 = f"""justice centers the moral obligation to address disproportionate impacts and systemic causes. 

        In climate, nature and development contexts, this means analysing climate change and the exploitation of natural resources as a political and ethical issue, recognising how climate change -- and the exploitation of natural resources more broadly -- impacts people, communities and countries differently and disproportionately, while benefiting others. Justice requires addressing structural inequities; including those affected and the most vulnerable in decision making; and recognising their well-being, as well as the value of different ways of knowing and being. To be considered justice-aligned, actions must aim to bring about a fairer, more inclusive world.

        Use the following inclusion/exclusion criteria, in addition to the definition:
        {criteria_q32}"""

    return description_q32



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