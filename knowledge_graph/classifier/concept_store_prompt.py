"""
Build LLM classifier system prompts from the Help: Labelling Principles page in the concept store.

The role preamble and the output-format rules stay in code because
the response parser depends on them and are not relevant for human labellers.
"""

from typing import Sequence

from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.wikibase import WikibaseSession

LABELLING_PRINCIPLES_PAGE = "Help:Labelling principles"

PROMPT_PREAMBLE = """
You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
These documents are mostly drawn from a climate and development context.
You will mark up references to concepts with XML tags.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>
"""

LABELLING_PRINCIPLES_INTRO = (
    "Apply the following labelling principles. They are the same principles our "
    "expert human labellers follow."
)

OUTPUT_FORMAT_INSTRUCTIONS = """
Output format (required, so that your answer can be parsed):

1. Surround each identified mention with <concept> tags.
2. If the passage contains multiple instances, each one should be tagged separately.
3. If the passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
4. If the entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
5. The input text must be reproduced exactly, down to the last character, even if this means typos or other minor formatting issues, only adding concept tags.
"""


def _escape_braces(text: str) -> str:
    """Escape braces so wiki text survives str.format() in LLMClassifierPrompt.format."""
    return text.replace("{", "{{").replace("}", "}}")


def build_system_prompt_template(principles_pages: Sequence[str]) -> str:
    """Combine the code-owned preamble and output rules with help page content (already fetched)."""
    body = "\n\n".join(_escape_braces(page) for page in principles_pages)
    return (
        PROMPT_PREAMBLE
        + f"\n{LABELLING_PRINCIPLES_INTRO}\n\n<labelling_principles>\n{body}\n</labelling_principles>\n"
        + OUTPUT_FORMAT_INSTRUCTIONS
    )


async def llm_prompt_from_concept_store_async(
    wikibase: WikibaseSession,
    page_titles: Sequence[str] = (LABELLING_PRINCIPLES_PAGE,),
    labelling_guidelines: str | None = None,
) -> LLMClassifierPrompt:
    """
    Build an LLMClassifierPrompt whose concept-agnostic labelling principles come from the concept store.

    :param page_titles: help pages to include, in order
    :param labelling_guidelines: optional concept-specific guidelines, appended as usual
    """
    pages = [await wikibase.get_help_page_text_async(t) for t in page_titles]
    return LLMClassifierPrompt(
        system_prompt_template=build_system_prompt_template(pages),
        labelling_guidelines=labelling_guidelines,
    )
