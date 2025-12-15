import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import nest_asyncio
    from dotenv import load_dotenv

    from knowledge_graph.classifier.large_language_model import LLMClassifier
    from knowledge_graph.cloud import AwsEnv
    from knowledge_graph.wikibase import WikibaseSession
    from scripts.get_concept import get_concept_async
    from scripts.train import main as train_cli

    # workaround for methods wrapped with @async_to_sync decorator
    nest_asyncio.apply()

    load_dotenv()
    return (
        AwsEnv,
        LLMClassifier,
        WikibaseSession,
        get_concept_async,
        mo,
        train_cli,
    )


@app.cell
def _(WikibaseSession):
    wikibase = WikibaseSession()
    WIKIBASE_ID = "Q1836"
    return WIKIBASE_ID, wikibase


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. get the concept
    """)
    return


@app.cell
async def _(WIKIBASE_ID, get_concept_async):
    concept = await get_concept_async(
        wikibase_id=WIKIBASE_ID,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )
    return (concept,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 common concept tasks
    """)
    return


@app.cell
def _(WIKIBASE_ID, wikibase):
    recursive_subconcepts = wikibase.get_recursive_has_subconcept_relationships(
        WIKIBASE_ID
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. create an `LLMClassifier`
    """)
    return


@app.cell
def _(LLMClassifier, concept):
    clf = LLMClassifier(concept, model_name="openrouter:openai/gpt-5")

    return (clf,)


@app.cell
def _(clf):
    print(clf.system_prompt)
    return


@app.cell
def _(clf):
    clf.concept.model_dump()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. evaluate an `LLMClassifier`
    """)
    return


@app.cell
def _():
    system_prompt_template = """
    You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
    These documents are mostly drawn from a climate and development context.
    You will mark up references to concepts with XML tags.

    First, carefully review the following description of the concept:

    <concept_description>
    {concept_description}
    </concept_description>

    Labelling guidelines for this concept:

    - Exclude general mentions of adaptation, if there's not a social or economic element
    - Mentions of economic adaptation should explicitly mention adjusting economic activities, e.g. individuals' livelihoods. Any activities e.g. production of crops that don't explicitly mention economic activities should not be identified as a mention.
    - Exclude texts that don't explicitly mention adaptation activities
    - Include mentions of activities that increase resilience of societies, e.g. social protection
    - Include mentions of protecting people from serious harm to the environment. Climate change is a harm to the environment.
    - Associate mentions of people, households, and residents in your definition of 'societal'.

    Instructions:

    1. Read through each passage carefully, thinking about the concept and different ways it can be used in documents.
    2. Identify any mentions of the concept, including references that are not included as an example, but which match the definition.
    3. Surround each identified mention with <concept> tags.
    4. If a passage contains multiple instances, each one should be tagged separately.
    5. If a passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
    6. If an entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag. Skip this step if you have tagged any concept mentions so far.
    7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
    8. Double check that you have tagged all mentions of the concept and that every tagged part is describing an actual mention of that concept."""
    return (system_prompt_template,)


@app.cell
def _(mo):
    mo.md(r"""
    TODO:

    - [ ] add direct subconcepts and their definitions as a part of the prompt
       - [ ] can remove "- Include mentions of activities that increase resilience of societies, e.g. social protection" from above after as it's in one of the definitions
    """)
    return


@app.cell
def _(AwsEnv, WIKIBASE_ID, system_prompt_template, train_cli):
    train_cli(
        wikibase_id=WIKIBASE_ID,
        track_and_upload=True,
        aws_env=AwsEnv.labs,
        classifier_type="LLMClassifier",
        classifier_override=[
            "model_name=openrouter:openai/gpt-5",
            f"system_prompt_template={system_prompt_template}",
        ],
        # concept_override=[f"definition={concept_definition}"],
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
