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
MODEL_NAME = "openrouter:google/gemini-3.1-pro-preview"

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
Use the following inclusion/exclusion criteria, in addition to the definition.

Your main task is to find positive examples; an EXCLUSION rule should never rule out whole passages, only specific phrases within that passage.

**THE NORMATIVE REQUIREMENT (Core Test)**
Before tagging, ask: Does this passage make an argument, either explicit or implicit, that is clearly linked to ethics, fairness or human rights, for improving the world or the position of vulnerable people?
ONLY tag parts of the text where the answer is yes. Remember you are a policy expert that is trained to see when and how abstract actions and arguments signal real commitment. 

**OVERALL GUIDELINE:** **Examples that FAIL the test:**
*  "Women in this sector often earn less than men" — states a fact about inequality without proposing a fix.  
*  "Land tenure insecurity leads to lower yields for female farmers" — impact analysis of a problem, not a justice solution.  
*  "Child labour is common in informal waste picking" — descriptive statement of fact.

**Examples that PASS the test:**
*  "The policy aims to close the unjust gender pay gap" — identifies an injustice and proposes an action to fix it.  
*  "We must ensure activities do not exacerbate existing inequalities" — a prescriptive safeguard to prevent injustice.  
* "Strategies to eliminate child labour and uphold human rights" — a prescriptive solution.  
* "We owe it to our children to ensure they can live fulfilling lives" — a normative ethical argument for intergenerational justice.


### **1. PROBLEMS VS. FIXING AND JUDGING**
To be tagged, a passage must not just *describe* an unfair situation; it MUST attempt to *fix it*, *judge* it ethically, or provide a framework to *prevent* it.
* **INCLUDE**: Justice-aligned actions, safeguards, solutions, and proposals. High-level policy commitments (e.g., "Mainstreaming a gender framework into project MRV") are active efforts to fix systemic gaps and should be included.  
* **EXCLUDE**: Passages that state an impact on a vulnerable group without providing a link to a remedial action or justice framing.

### **2. ETHICAL ARGUMENTS & RIGHTS-BASED REASONING**
* **INCLUDE**: Passages stating why justice is important or why actions should be taken in a just, fair, or equitable way.  
* **INCLUDE**: Rights-based reasoning where the text relies on "The Right to [X]" or "Rights of [Group]" as a tool to argue for social change, climate action, or the protection of vulnerable people.

### **3. JUSTICE-ALIGNED VS. GENERIC ACTION**
Avoid tagging buzzwords unless they are specifically qualified by justice concepts.
* **INCLUDE**: phrases with justice-signalling qualifiers such as "ensure **Equal** representation" or "**Equitable** access to finance."  
* **SCRUTINIZE**: Vague language, buzzwords and tokenistic mentions, like "Empowering communities" or "Inclusive by design." Tag ONLY if the phrase is made credible by additional information (e.g. vulnerable groups, pollicy action).

### **4. HOLISTIC JUSTICE**
* **INCLUDE**: Passages that describe a specific type of justice. Tag any phrase matching the following definitions:  
  a) Recognition Justice: {recognition.definition}  
  b) Procedural Justice: {procedural.definition}  
  c) Distributive Justice: {distributive.definition}  
  d) Transformative Justice: {transformative.definition}  
* **INCLUDE**: Ethical discussions of the root causes of inequality and proposals or arguments to fix these root causes.

### **5. INSTITUTIONAL TITLES & CONTEXT**
* **INCLUDE**: Titles of policies, departments, or priority areas that name justice frameworks or targeted benefits (e.g., "Ministry GESI Action Plan," "Just Transition Priority Areas," "Regional Rights Resource Team"). These labels often represent the core substance of the text.  
* **ACRONYMS**: Ambiguous acronyms (like "JET") should be interpreted within a climate-, nature-, and development context. Assume they are justice-relevant if the context points toward a "Just Transition."

### **6. PRIORITIZE EXPLICIT KEYWORDS**
* **INCLUDE**: Explicit mentions of "Intergenerational justice," "Climate Ethics," "Restorative Justice," or "Climate Justice."  
* **INCLUDE**: "Gender," "Vulnerable," or "Indigenous" ONLY when they are part of a phrase fulfilling the Normative Requirement (e.g., a goal to "double the productivity of indigenous farmers").  
* **EXCLUDE**: Criminal justice or legal systems unless they are explicitly relevant to climate/environmental rights or transition impacts.

### **7. SHORT PHRASES AND CITATIONS***
*   **SCRUTINIZE** titles, headers, proper nouns, or names of committees/documents. These are often too generic to be included, but can be included if they point point to justice-relevant policy or explicitly mention justice-aligned content. 
*   **Exclude** citations or references generally. Explicit mentions of core justice concepts can sometimes be included, but not if they are generic (e.g., "As discussed in the 2021 Justice Reforms Report...").

### **7. AMBIGUITY CHECK**
* **If the text identifies a justice-relevant barrier (e.g., "discriminatory norms prevent participation"):** Tag it if it is part of a strategy intended to address that barrier.  
* **If the text names a justice-relevant goal (e.g., "Objective: Allocate 40% of funds to female-headed households"):** Tag it.  
* **If the text does not match any criteria, return to the Core Test and the definition.**
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
        max_negative_proportion=0.3,
        corpus_types_exclude=["Litigation"],
        dataset_name=dataset_name,
        max_size_to_sample_from=max_size_to_sample_from,
        track_and_upload=track_and_upload,
        concept_override=concept_overrides,
    )


if __name__ == "__main__":
    app()
