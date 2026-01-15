"""Train an LLM for Q912 distributive justice"""

from rich.console import Console

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.train import main as train_cli

MODEL_NAME = "openrouter:google/gemini-3-pro-preview"


def train_distributive_justice() -> None:
    """Run training"""

    console = Console()

    console.log("Connecting to Wikibase")
    wikibase = WikibaseSession()

    console.log("Fetching 3 related justice concepts from Wikibase")
    distributive = wikibase.get_concept(WikibaseID("Q911"))
    transformative = wikibase.get_concept(WikibaseID("Q1730"))
    recognition = wikibase.get_concept(WikibaseID("Q1731"))

    criteria = f"""
		### THE POWER TEST
		Before tagging, ask: Does this passage describe a process that helps marginalized or impacted groups to influence, challenge, or shape a decision? 
		ONLY tag if the answer is yes. In addition:
		
		### 1.  MEANINGFUL PARTICIPATION vs. GENERIC MENTIONS
		- **INCLUDE**: Specific, named participatory methods (e.g. co-design, Participatory Land Use Planning) as long as impacted groups have a meaningful role.
		- **NOTE**: Be careful around generic inclusion methods (e.g. workshops, consultation, engagement). Especially if they explicitly mention involvement of a vulnerable group, it can pass the POWER TEST, but without mention of a group or goal, they can mean one-way information session which should be excluded. 
		- **EXCLUDE**: Tokenistic inclusion where consultation is a checkbox. 
		- **EXCLUDE**: Mentions of groups receiving things (e.g. money, training) without a say in the management of those things (distribution, not participation). 
		
		### 2. ACCESSIBLE JUSTICE vs. ADMINISTRATIVE LAW
		- **INCLUDE**: Grievance mechanisms, accountability channels, or rights to appeal ONLY if they are accessible to local communities or the vulnerable.
		- **EXCLUDE**: Standard administrative law boilerplate. These are general regulatory functions, not procedural justice.
		
		### 3. EMPOWERING TRANSPARENCY vs. COMPLIANCE
		- **INCLUDE**: Transparency processes that builds community agency or explicitly mentions accountability to impacted/vulnerable groups.
		- **EXCLUDE**: Technical or bureaucratic transparency (e.g. publishing emission inventories, financial audits, standard MRV). These are for donors/regulators/other countries, not for justice.
		
		### 4. FAIRNESS vs. PROTECTIVE SAFEGUARDS
		- **INCLUDE**: Include measures which meaningfully improve the fairness of decision making, such as special protections and procedures for vulnerable groups.
		- **EXCLUDE**: General safety protocols or confidentiality measures. While vital for safety, these are safeguards, not procedural justice (decision-making power).
		
		### 5. PROCEDURAL JUSTICE SOLUTIONS vs. PROBLEM DESCRIPTIONS
		- **INCLUDE**: Passages that describe a *lack* of procedural justice as a justice problem or that suggest fair decision-making as a potential solution.
		- **EXCLUDE**: General descriptions of injustice or unjust representation without a mention of procedural remedies.	
		
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
		"""

    instructions = """
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

    new_concept_definition = """Procedural justice means ensuring that decision making is fair and inclusive, emphasizing the agency and influence of vulnerable groups and those, giving them power to change processes that affect their lives."""
    new_concept_description = f"""Nominally, procedural justice is focussed on fair decisions, but in practice, this means changing the structural mechanisms of influence.

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

    strong_persona_system_prompt_template = (
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
        + instructions
    )

    console.print("Training model with strong persona template")
    train_cli(
        wikibase_id=WikibaseID("Q912"),
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            f"model_name={MODEL_NAME}",
            f"system_prompt_template={strong_persona_system_prompt_template}",
        ],
        concept_override=[
            f"definition={new_concept_definition}",
            f"description={new_concept_description}",
        ],
    )

    console.print("Training model with default template")
    train_cli(
        wikibase_id=WikibaseID("Q912"),
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            f"model_name={MODEL_NAME}",
        ],
        concept_override=[
            f"definition={new_concept_definition}",
            f"description={new_concept_description}",
        ],
    )


if __name__ == "__main__":
    train_distributive_justice()
