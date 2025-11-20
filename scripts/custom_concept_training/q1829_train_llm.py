"""
Train financial flow (Q1829) classifier with custom definition and prompt.

This script is needed because the custom definition contains annotation guidelines,
which exceed the length allowed in Wikibase
"""

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from scripts.train import main as train_cli


def train_financial_flow() -> None:
    """Run training"""

    concept_definition = """A finance flow is an economic flow that reflects the creation, transformation, exchange, transfer, or extinction of economic value
    and involves changes in ownership of goods and/or financial assets, the provision of services, or the provision of labor and capital.
    ideally, a financial flow describes four elements:
    1. the source (who is sending the financial asset, such as a bank or organisation);
    2. the financial instrument or mechanism (how it is being sent, such as a grant, loan or a subsidy);
    3. the use or destination (the purpose for which the asset will be used, which is often expressed as the recipient organisation or
    their sectoral categorisation); and
    4. the value (which can, but does not need to be, expressed in monetary terms directly).
    It is acceptable if not all of these elements are present, but where they are, all of these elements should be considered part
    of the same financial flow.

    Guidelines for annotation are:
    
        - Include when the text describes money or financial assets moving (e.g. payments, loans, investments, disbursements, repayments).
        - Include both one-off transactions and ongoing streams (e.g. monthly payments, yearly disbursements).
        - Include in the same label all relevant elements of the flow (if given): who is paying, who is receiving, the purpose, the amount and the timeframe, as well as any crucial details, such as legal conditions or interest payable. However, always ensure such statements include the flow itself; do not tag if the text only specifies conditions.
        - Include financial commitments, as well as disbursements, as long as commitments are firm, such as budgets and pledges. 
        - Exclude when the text only states amounts held, owed, or valued at a point in time (these are stocks, not flows).
        - Exclude metaphorical/non-financial uses of "flow" (e.g. "flow of information").
        - Exclude hypothetical flows and financial discussions which are not tied to concrete financial commitments.
        - Exclude financial flows that are outside the real economy, which means excluding factors like operational expenses or investments made into financial markets. Note, however, that flows such as development assistance or payments to/from climate funds are to be included. 
        - Exclude discussions on the need for finance or finance flows, as well as discussions- or statistics on the economy or performance of a company.
"""

    system_prompt_template = """
    You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
    These documents are mostly drawn from a climate and development context.
    You will mark up references to concepts with XML tags.

    First, carefully review the following description of the concept:

    <concept_description>
    {concept_description}
    </concept_description>

    Instructions:

    1. Read through each passage carefully, thinking about the concept and different ways it can be used in documents.
    2. Identify any mentions of the concept, including references that are not included explicitly as examples, but which do match the definition and guidelines.
    3. Surround each identified mention with <concept> tags.
    4. If a passage contains multiple instances, each one should be tagged separately.
    5. If a passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
    6. If an entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
    7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
    8. Double check that you have tagged all financial flows and that every tagged part is describing an actual financial flow, including the source, destination, instrument and value, if any is given.
    """

    train_cli(
        wikibase_id=WikibaseID("Q1829"),
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            "model_name=gpt-5",
            f"system_prompt_template={system_prompt_template}",
        ],
        concept_override=[f"definition={concept_definition}"],
    )


if __name__ == "__main__":
    train_financial_flow()
