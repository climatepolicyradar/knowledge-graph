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
		Before tagging, ask: Does this passage §ibe a process that helps marginalized or impacted groups to influence, challenge, or shape a decision? 
		ONLY tag if the answer is yes. In addition:
		
		### 1.  MEANINGFUL PARTICIPATION vs. GENERIC MENTIONS
		- **INCLUDE**: Specific, named participatory methods (e.g. co-design, Participatory Land Use Planning) as long as impacted groups have a meaningful role.
		- **NOTE**: Be especially careful around generic inclusion methods (e.g. workshops, consultation, engagement). EXCLUDE generic mentions that lack detail on who is involved or how their input is used, but INCLUDE if it passes the POWER TEST. 
		- **EXCLUDE**: Tokenistic inclusion where consultation is a checkbox. Also EXCLUDE mentions of groups receiving things (money, training) without a say in the management of those things.
		
		### 2. ACCESSIBLE JUSTICE vs. ADMINISTRATIVE LAW
		- **INCLUDE**: Grievance mechanisms, accountability channels, or rights to appeal ONLY if they are explicitly described as accessible to local communities or the vulnerable.
		- **EXCLUDE**: Standard administrative law boilerplate. These are general regulatory functions, not procedural justice.
		
		### 3. EMPOWERING TRANSPARENCY vs. COMPLIANCE
		- **INCLUDE**: Transparency that builds community agency or explicitly mentions accountability to impacted/vulnerable groups.
		- **EXCLUDE**: Technical or bureaucratic transparency (e.g. publishing emission inventories, financial audits, MRV). These are for donors/regulators, not for justice.
		
		### 4. FAIRNESS vs. PROTECTIVE SAFEGUARDS
		- **INCLUDE**: Include measures which meaningfully improve the fairness of decision making, such as special protections and procedures for vulnerable groups.
		- **EXCLUDE**: General safety protocols or confidentiality measures. While vital for safety, these are safeguards, not procedural justice (decision-making power).
		
		### 5. PROCEDURAL JUSTICE SOLUTIONS vs. PROBLEM DESCRIPTIONS
		- **INCLUDE**: Passages that describe a *lack* of procedural justice as a problem or suggest fair decision-making as a potential solution.
		- **EXCLUDE**: Descriptions of unjust situations, including unjust representation, if there is no explicit mention of the need for procedural justice or procedural justice solutions.
		
		### 6. JUSTICE TYPE DIFFERENTIATION
		- **EXCLUDE**: generic discussions of justice, or other types of justice, such as the Justice System.
		- **EXCLUDE** passages that fit better under these definitions:
            - Recognition Justice: {recognition.definition}
            - Transformative Justice: {transformative.definition}
            - Distributive Justice: {distributive.definition}
		"""

    instructions = """
    Instructions:
        1. Read through the passage carefully, thinking about the concept and different ways it is used in documents.
        2. Identify any mentions of the concept, including direct references and indirect descriptions of the concept.
        3. Surround each identified mention with <concept> tags.
        4. If the passage contains multiple instances, each one should be tagged separately.
        5. If the passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
        6. If the entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
        7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
        8. Double check that you have tagged all instances of climate justice, according to the provided definition, and that every tagged part contains enough information to show why this is relevant.
    """

    new_concept_definition = """Procedural justice means ensuring that decision making is fair and inclusive, emphasizing the agency and influence of vulnerable groups and those, giving them power to change processes that affect their lives."""
    new_concept_description = f"""Nominally, procedural justice is focussed on fair decisions, but in practice, this means changing the structural mechanisms of influence.
        
        This sets a high bar: 
        
        It is defined by four pillars:
            1. Voice: Impacted groups have a formal seat at the table and participate actively and meaningfully in decision making. If decisions go wrong, there are accessible, community-specific channels to appeal or seek redress.
            2. Respect: Treating all participants with respect and dignity.
            3. Neutrality: Decisions are unbiased and guided by unbiased transparent reasoning. 
            4. Transparency: Decision-making is transparent enough that external groups can audit and challenge the logic behind a choice.
            
        These four elements may not always be mentioned concurrently, but they all should contribute to improving the real agency of those most impacted and legitimacy of decision-making.
        
        Use the following inclusion/exclusion criteria to distinguish between genuine procedural power and standard administrative bureaucracy:
        {criteria}
        """

    strong_persona_system_prompt_template = (
        """
        You are a specialist analyst, tasked with identifying mentions of 
        concepts in policy documents. You are also a climate justice activist 
        from a climate-vulnerable community and incorporate your lived experience 
        into your analysis. You have an expert understanding of legal theory,
        postcolonialism, feminist and queer theory, as well as protest movements.

        You are trying to construct a database of relevant passages that can help
        climate justice activists and policymakers make more informed decisions on climate justice.
        This means you want to show a broad cross section, but always remain critical of the source material.  
        Your goal is to identify passages which discuss marginalized groups gaining meaningful power
        and structural changes in decision-making processes that are justice-aligned.
        
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
