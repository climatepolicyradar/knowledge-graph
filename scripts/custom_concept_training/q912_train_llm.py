"""Train an LLM for Q912 distributive justice"""

from rich.console import Console

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession
from scripts.train import main as train_cli

MODEL_NAME = "openrouter:openai/gpt-5"


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
            - A passage can ONLY be included (i.e. tagged)if it is arguing for- or concretely describing fair, inclusive decision-making. 
            - Focus  particularly on  passages which would meaningfully benefit the agency of vulnerable groups.
            - Include also if procedural justice is described, but not explicitly mentioned.
            - Include actions that would meaningfully improve transparency, accountability mechanisms, or the right to appeal (correctability).
            - Include passages that describe procedural justice as a potential solution, or which mention a lack of procedural justice explicitly as a problem. 
            - Include passages that describe fair and just outcomes (distributive justice) only if they mention inclusive decision-making as a means to achieve this. In this case, place tags as close to the distributive phrases as possible.

            - Exclude (i.e. no tags) if there is no clear process element or argument described.
            - Exclude criminal justice and discussions of the justice system, unless they are relevant to climate, development or nature (interpreted broadly). 
            - Exclude passages that are about generic justice (i.e. there is no clear process element). 
            - Exclude passages that appear to be greenwashing, climate misinformation or which are too small to contribute meaningfully to justice. Be particularly mindful of tokenistic inclusion that does not support the agency of vulnerable groups, such as instances where consultation and stakeholder engagement are a checkbox exercise, or where there is no information on who is involved, nor the goals . 
            - Exclude mentions of vulnerable groups receiving things (training, money, food, ...) without any mention of them having a say in how those things are managed (i.e. no procedural invovlement).
            - Exclude general descriptions of injustices, including for example general statistics. These can only be included if they are clearly within a justice context and relevant to decision making. 
            - Exclude all passages that better fit other types of climate justice, especially recognition, transformative and distributive justice.
            - For those other types of justice, use the following definitions: 
            - Exclude: {recognition.definition};
            - Exclude: {transformative.definition}; 
            - Exclude: {distributive.definition}.
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

    new_concept_definition = (
        """Procedural justice means ensuring that decision making is fair and inclusive emphasizing the agency and influence of vulnerable groups and those most impacted.""",
    )
    new_concept_description = f"""Procedural justice means creating fair, inclusive and transparent decision-making procedures, which includes 4 elements: 
        Voice: Individuals are given a chance to express their concerns and participate in decision-making processes by telling their side of the story
        Respect: All individuals are treated with dignity and respect
        Neutrality: Decisions are unbiased and guided by consistent and transparent reasoning
        Trustworthiness: Decision-makers convey trustworthy motives and concern about the well-being of those impacted by their decisions
        In climate- and development contexts, this means including those impacted by climate change and the transition to a regenerative economy into decision making processes, making sure their voice is heard and power is given to those most affected.                    
        
        Use the following inclusion/exclusion criteria, in addition to the definition:
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
