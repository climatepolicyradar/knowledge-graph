"""
Train and sample procedural justice (Q912) classifier with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase.

Usage:
    uv run scripts/custom_concept_training/q912_train_llm.py train
    uv run scripts/custom_concept_training/q912_train_llm.py sample
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

WIKIBASE_ID = WikibaseID("Q912")
MODEL_NAME = "openrouter:google/gemini-3-pro-preview"

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
    transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria = f"""
		### THE POWER TEST
        Before tagging, ask: Does this passage describe a process that helps marginalized or impacted groups to influence, challenge, or shape a decision?
        ONLY tag if the answer is yes.

        **Examples that FAIL the power test:**
        - "it is necessary to empower the most vulnerable" — too vague, no mechanism described
        - "civil society must hold governments to account" — no procedural element
        - "increased communication with energy users" — interaction ≠ influence
        - "stakeholders will be consulted during planning and monitoring" — unclear who stakeholders are, nor if they have power to shape decisions

        **Examples that PASS the power test:**
        - "subsistance farmers took part in workshops on land tenure rights, including joint management" — organising of an impacted group towards inclusive decision-making
        - "Indigenous Peoples have mandated representation in the project oversight committee" - meaningful procedural safeguards
        - "promoted PLP in 40 rural communities" - in this context, PLP is likely participatory land use planning, a strong method with a named vulnerable group
        - "the national committee's proposal reflects input from over 30,000 citizens, trade unions and advisory bodies" — large-scale public input sollicited by a powerful institution suggests meaningful change
        - "co-design with local stakeholders through sectoral workshops" — strong participatory method with stakeholders
        - "women-led FGDs resulted in" — Focus Group Discussions are not very strong, but with a named group and suggestion that outcomes were taken up

        More formally, use the the following criteria to tag:

		### 1.  MEANINGFUL PARTICIPATION vs. GENERIC MENTIONS
		- **INCLUDE**: Specific, named participatory methods (e.g. co-design, Participatory Land Use Planning) as long as impacted groups have a meaningful role.
        - **INCLUDE**: Large-scale public input processes where significant numbers of citizens, communities, or Indigenous peoples contributed to policy development.
        - **INCLUDE**: Processes described as building "consensus" through multi-stakeholder engagement — consensus-building implies negotiation and influence, not one-way information.
		- **NOTE**: Be careful around generic inclusion methods (e.g. workshops, consultation, engagement):
            a)Especially if they explicitly mention involvement of a vulnerable group, it can pass the POWER TEST.
            b) Without mention of a group or goal, such generic languages suggests a one-way information session which should be excluded.
		- **EXCLUDE**: Tokenistic inclusion where consultation is a checkbox.
		- **EXCLUDE**: Mentions of groups receiving things (e.g. money, training) without a say in the management of those things (distribution, not participation).

		### 2. ACCESSIBLE JUSTICE vs. ADMINISTRATIVE LAW
		- **INCLUDE**: Grievance mechanisms, accountability channels, or rights to appeal ONLY if they are accessible to local communities or the vulnerable.
		- **EXCLUDE**: Standard administrative law boilerplate. These are general regulatory functions, not procedural justice.

		### 3. EMPOWERING TRANSPARENCY vs. COMPLIANCE
		- **INCLUDE**: Transparency processes that builds community agency. Also include transparency processes that explicitly mention accountability to impacted/vulnerable groups.
		- **EXCLUDE**: Technical or bureaucratic transparency (e.g. publishing emission inventories, financial audits, standard MRV). These are for donors/regulators/other countries, not for justice.

		### 4. FAIRNESS vs. PROTECTIVE SAFEGUARDS
		- **INCLUDE**: Include measures which meaningfully improve the fairness of decision making, such as special protections and procedures for vulnerable groups.
		- **EXCLUDE**: General safety protocols or confidentiality measures. While vital for safety, these are safeguards, not procedural justice (decision-making power).

        ### 5. PROCEDURAL JUSTICE SOLUTIONS vs. PROBLEM DESCRIPTIONS
        - **INCLUDE**: Passages that describe a *lack* of procedural justice as a justice problem or that suggest fair decision-making as a potential solution.
        - **EXCLUDE**: General descriptions of injustice or unjust representation without a mention of procedural remedies.
        - **EXCLUDE**: Descriptions of barriers TO participation (e.g. "discriminatory norms prevent women from decision-making", "fail to take into account needs of vulnerable groups"). These describe problems, not solutions. Only tag if the passage also describes mechanisms to ADDRESS these barriers.

		### 6. INSTITUTIONAL POWER & MAINSTREAMING
		- **INCLUDE**: High-level mandates and formal commitments from those in power to give decision-making power to vulnerable stakeholders. These represent institutional commitments to justice.
		- **NOTE**: If a justice-relevant topic is being 'mainstreamed', the nominal meaning is an institution-wide consideration for the topic which should be included, unless it is very clearly used only as a buzzword or a vague goal.
		- **INCLUDE**: Decentralization/Joint-Managemen should be tagged, but only if power is (likely) being moved closer to the community.

		### 6. JUSTICE TYPE DIFFERENTIATION
		- **EXCLUDE**: generic discussions of justice, or other types of justice, such as the Justice System.
		- **EXCLUDE** passages that fit better under these definitions:
            - Recognition Justice: {recognition.definition}
            - Transformative Justice: {transformative.definition}
            - Distributive Justice: {distributive.definition}

        ### 7. CONTEXT REQUIREMENT
        - **EXCLUDE**: Single abstract words (e.g. "Democratization", "Accountability") without any context.
        - **NOTE**: When in doubt about short phrases with no means to gauge the depth of involvement nor the importance of the process, ask: does this name a specific process (consultation, FGD, workshop, input gathering) AND at least one impacted group or justice goal? If yes, include it.

        ### 8. HEADINGS, DOCUMENT TITLES AND REFERENCES
        - **INCLUDE**: Headings, document titles, report names, or publication references that explicitly name participatory processes (e.g. "Lessons from the field: Including Women in Community Forestry").
        - **NOTE**: Even in reference lists or citations, tag the participation-relevant portion of the title. The title indicates the document substantively addresses procedural justice.
        """

    return f"""Nominally, procedural justice is focussed on fair decisions, but in practice, this means changing the structural mechanisms of influence.

        This sets a high bar:

        It is defined by four pillars:
            1. Voice: Impacted groups have a formal seat at the table and participate actively and meaningfully in decision making. If decisions go wrong, there are accessible, community-specific channels to appeal or seek redress.
            2. Respect: Treating all participants with respect and dignity.
            3. Neutrality: Decisions are unbiased and guided by unbiased transparent reasoning.
            4. Transparency: Decision-making is transparent enough that external groups can audit and challenge the logic behind a choice.

        These four elements are all necessary and build on each other; though they may not always be mentioned concurrently, they contribute to improving the real agency of those most impacted and legitimacy of decision-making.

        Use the following inclusion/exclusion criteria to distinguish between genuine procedural power and standard administrative bureaucracy:
        {criteria}
        """


def get_system_prompt_template() -> str:
    """
    Build the strong persona system prompt template.

    :returns: The system prompt template with instructions.
    :rtype: str
    """
    return (
        """
        You are a specialist analyst and a climate justice activist from a
        climate-vulnerable community. You combine expert policy knowledge,
        with a critical perspective on power dynamics. This is informed by
        your practical lived experience, as well as legal- and decolonial theory.

        Your goal is to identify passages that demonstrate marginalized
        groups gaining meaningful influence or agency in decision-making.
        You are critical of tokenism, but you are alert to the ways procedural
        justice is formalized in policy language. This means you recognise that
        policy documents use terms like mainstreaming and decentralization in
        varied ways: it can be empty jargon and checkbox exercises, but it
        can also be a meaningful step towards real influence in projects
        and society.


        You will mark up references to concepts with XML tags.

        First, carefully review the following description of the concept:

        <concept_description>
        {concept_description}
        </concept_description>
        """
        + INSTRUCTIONS
    )


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
    system_prompt_template = get_system_prompt_template()

    console.print("Training model with strong persona template")
    train_cli(
        wikibase_id=WIKIBASE_ID,
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            f"model_name={MODEL_NAME}",
            f"system_prompt_template={system_prompt_template}",
        ],
        concept_override=concept_overrides,
    )

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
