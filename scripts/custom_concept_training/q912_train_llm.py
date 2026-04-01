"""
Train and sample procedural justice (Q912) classifier with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q912_train_llm.py train
    uv run scripts/custom_concept_training/q912_train_llm.py sample
"""

import asyncio

import typer
from rich.console import Console

from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.sample import main as sample_cli
from scripts.train import run_training

app = typer.Typer()
console = Console()

WIKIBASE_ID = WikibaseID("Q912")
MODEL_NAME = "openrouter:google/gemini-3.1-pro-preview"

CONCEPT_DEFINITION = """Procedural justice means ensuring that decision making is fair and inclusive, emphasizing the agency and influence of vulnerable groups and those, giving them power to change processes that affect their lives."""

INSTRUCTIONS = """
    Instructions:
        1. Read through the passage carefully, thinking about the concept and different ways it is used in documents, including acronyms, jargon and global differences.
        2. Identify any mentions of the concept, including direct references and indirect descriptions of the concept which match the definition.
        3. Surround each identified mention with <concept> tags.
        4. If the passage contains multiple instances, each one should be tagged separately.
        5. If the passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
        6. If the entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
        7. The input text must be reproduced exactly, down to the last character, even including typos if relevant, only adding concept tags.
        8. Double check that you have tagged all instances of procedural justice according to the provided definition, and that every tagged part contains enough information to show why this is relevant.
    """


def get_labelling_guidelines() -> str:
    """
    Build the labelling guidelines by fetching related justice concepts from Wikibase.

    :returns: The labelling guidelines with dynamic criteria for procedural justice.
    :rtype: str
    """
    console.log("Connecting to Wikibase")
    wikibase = WikibaseSession()

    console.log("Fetching 3 related justice concepts from Wikibase")
    distributive = wikibase.get_concept(WikibaseID("Q911"))
    # transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria = f"""
Use the following inclusion/exclusion criteria, in addition to the definition.
These are guidelines meant to help you disambiguate. 
Overall, your main task is to find positive examples. INCLUSION therefore generally trumps EXCLUSION, and an EXCLUSION rule should never rule out whole passages, only specific phrases within that passage.

**Core test:** Procedural justice is the *active* inclusion of marginalized or impacted groups in decision-making, allowing them to influence, challenge, or shape outcomes. It is not merely the existence of standard bureaucracy or one-way communication.
You are looking for phrases that pass this test, meaning they contain a mechanism or argument for fairer participation, keeping in mind that you are a policy expert who is trained to read between the lines.

**Examples that FAIL the test:**
- "it is necessary to empower the most vulnerable" — too vague, just "empower" is not a procedural mechanism.
- "civil society must hold governments to account" — generic statement, no concrete procedural element.
- "notifying the public of the new energy rates" — passive, one-way information and part of standard business practice.
- "stakeholders will be consulted" — unclear who stakeholders are, nor if they are an impacted/vulnerable group.

**Examples that PASS the test:**  
- "subsistence farmers took part in workshops on land tenure rights, including joint management" — organizing an impacted group towards inclusive decision-making.  
- "Indigenous Peoples have mandated representation in the project oversight committee" - meaningful procedural safeguards.  
- "the national committee's proposal reflects input from over 30,000 citizens and trade unions" — large-scale public input solicited by a powerful institution suggests meaningful change.  
- "co-design with local stakeholders through sectoral workshops" — strong participatory method with impacted groups.

### **1. THE INFLUENCE/POWER REQUIREMENT**
To be tagged, the passage must do more than mention "transparency" or "communication". It must contain a **mechanism**—either a specific participatory process, an accountability channel, or an institutional mandate.
* **Action:** INCLUDE if the text describes a process where impacted groups, communities, or their representatives provide input, feedback, or engagement that shapes outcomes (e.g., consultations, workshops). Input gathering is an exercise of power.  
* **Argument:** INCLUDE if the text claims that decision-making *ought* to be more inclusive, transparent, or accountable to vulnerable groups based on fairness.

### **2. INCLUSION CRITERIA**
Tag the passage if it matches any of the following:

**A. Meaningful Participation & Consultation**
* **INCLUDE** specific, named participatory methods (e.g., co-design, Participatory Land Use Planning, FGDs) where impacted groups have a role.  
* **INCLUDE** consultations, workshops, or "convening stakeholders" whenever the participants are **impacted groups** (workers, local communities, women) or their representatives. In policy language, these are the primary vehicles for procedural influence.  
* **INCLUDE** large-scale public input processes or consensus-building through multi-stakeholder engagement.

**B. Accessible Accountability & Grievance**
* **INCLUDE** grievance mechanisms (GRMs), accountability channels, or rights to appeal ONLY if they are explicitly accessible to local communities or the vulnerable.  
* **INCLUDE** transparency processes that build community agency or explicitly mention accountability to impacted/vulnerable groups.

**C. Institutional Power & Coordination**
* **INCLUDE** high-level mandates and formal commitments to "mainstream" justice or give decision-making power to vulnerable stakeholders.  
* **INCLUDE** Decentralization, Joint-Management, or Co-management if power is being moved closer to the community.  
* **INCLUDE** formal inter-governmental coordination (e.g., "convening provincial and national departments to agree on transition requirements") as this represents the procedural structure of justice in a policy context.

**D. Addressing Barriers**
* **INCLUDE** passages that describe a *lack* of procedural justice (e.g., "discriminatory norms prevent women from decision-making") IF the text is part of a strategy/policy document and implies or describes mechanisms to address these barriers. Identifying the gap is often the first step in a procedural remedy.

### **3. EXCLUSION CRITERIA**

Do **NOT** tag the passage if it falls into these categories:

**A. Passive Information & Compliance**
* **Exclude** purely passive/one-way communication where the public only receives info (e.g., "broadcasting information", "publishing reports").  
* **Exclude** technical or bureaucratic transparency (e.g., publishing emission inventories, financial audits, standard MRV) UNLESS the text mentions a stakeholder consultation process to design/review them.

**B. Generic Administrative Law & Safeguards**
* **Exclude** standard administrative law boilerplate and regulatory functions that do not explicitly mention public/vulnerable input. *(Note: Regulatory bodies providing 'transparent and impartial' oversight to the public ARE providing procedural justice and should be included).*  
* **Exclude** general safety protocols or confidentiality measures. These are protective safeguards, not decision-making power.

**C. Nominal Mentions & Abstract Terms**
* **Exclude** single abstract words (e.g., "Democratization", "Accountability") without context.  
* **Exclude** generic inclusion methods (e.g., "workshops", "meetings") if there is NO mention of an impacted group or justice goal.  
* **NOTE:** Project/policy titles or report headings CAN be included if they explicitly name participatory processes (e.g., "Lessons from the field: Including Women in Community Forestry").

**D. Resource Distribution (Other Justice Types)**
* **Exclude** mentions of groups receiving things (e.g., money, training, capacity building) without a say in the management of those things (this is Distributive, not Procedural).  
* **Exclude** passages that fit better under:  
  - Distributive Justice: {distributive.definition}
  - Recognition Justice: {recognition.definition}

### **4. AMBIGUITY CHECK**
* **If the text implies "engagement" but is vague:** Ask, "Does this name a specific process (consultation, FGD, input gathering) AND at least one impacted group or justice goal?" If yes, **INCLUDE**.  
* **If the text identifies a barrier to participation:** Ask, "Is this part of an argument or strategy intent on fixing it?" If yes, **INCLUDE**.  
* **If the text does not match any of the criteria, go back to the definition and the core test.**  
"""

    return INSTRUCTIONS + "\n\n" + criteria


def get_concept_description() -> str:
    """
    Build the concept description without the detailed criteria.

    :returns: The concept description focused on the four pillars of procedural justice.
    :rtype: str
    """
    return """Nominally, procedural justice is focussed on fair decisions, but in practice, this means changing the structural mechanisms of influence.

This sets a high bar:

It is defined by four pillars:
    1. Voice: Impacted groups have a formal seat at the table and participate actively and meaningfully in decision making. If decisions go wrong, there are accessible, community-specific channels to appeal or seek redress.
    2. Respect: Treating all participants with respect and dignity.
    3. Neutrality: Decisions are unbiased and guided by unbiased transparent reasoning.
    4. Transparency: Decision-making is transparent enough that external groups can audit and challenge the logic behind a choice.

These four elements are all necessary and build on each other; though they may not always be mentioned concurrently, they contribute to improving the real agency of those most impacted and legitimacy of decision-making.
"""


def get_concept_overrides() -> dict[str, str]:
    """
    Get the concept override dict for training and sampling.

    :returns: Dictionary of concept property overrides.
    :rtype: dict[str, str]
    """
    concept_description = get_concept_description()
    return {
        "definition": CONCEPT_DEFINITION,
        "description": concept_description,
    }


@app.command()
def train() -> None:
    """Run training for the procedural justice classifier."""
    concept_overrides = get_concept_overrides()
    labelling_guidelines = get_labelling_guidelines()

    from knowledge_graph.classifier.large_language_model import DEFAULT_SYSTEM_PROMPT

    default_prompt = LLMClassifierPrompt(
        system_prompt_template=DEFAULT_SYSTEM_PROMPT,
        labelling_guidelines=labelling_guidelines,
    )

    console.print("Training model with default template")
    asyncio.run(
        run_training(
            wikibase_id=WIKIBASE_ID,
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            classifier_type="LLMClassifier",
            classifier_kwargs={
                "model_name": MODEL_NAME,
                "system_prompt_template": default_prompt,
            },
            concept_overrides=concept_overrides,
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
    """Sample passages for the procedural justice concept with custom definition."""
    concept_overrides = get_concept_overrides()
    # Convert dict to list of key=value strings for sample_cli
    concept_override_strings = [f"{k}={v}" for k, v in concept_overrides.items()]
    sample_cli(
        wikibase_id=WIKIBASE_ID,
        sample_size=sample_size,
        min_negative_proportion=0.1,
        dataset_name=dataset_name,
        max_size_to_sample_from=max_size_to_sample_from,
        track_and_upload=track_and_upload,
        concept_override=concept_override_strings,
    )


if __name__ == "__main__":
    app()
