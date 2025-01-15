from itertools import chain
import os
import re

from anthropic import Anthropic
from src.concept import Concept
from src.wikibase import WikibaseSession
from rich import print
from tqdm import tqdm
from requests import get


model = "claude-3-5-haiku-20241022"
llm =  Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_correctable_concepts() -> dict[Concept, dict[str, list[str]]]:
    session = WikibaseSession()

    concepts = session.get_concepts(limit=500)

    correctable_concepts = {}

    for c in tqdm(concepts):
        if (len(str(c.definition)) < 20):
            statement_to_references = session.get_statement_to_references(c.wikibase_id)
            if statement_to_references:
                correctable_concepts[c] = statement_to_references
    return correctable_concepts


def document_filter(document: str, keywords: list[str]) -> str:
    matching_line_indices = []
    lines = document.split("\\n")
    for idx, line in enumerate(lines):
        if re.match(r"\b(?:" + "|".join(keywords) + r")\b", line, re.IGNORECASE):
            matching_line_indices.append(idx)

    values = []
    for index in matching_line_indices:
        values.extend(list(range(max(index - 5, 0), min(index + 5, len(lines)))))

    values = sorted(list(set(values)))
    
    return "\n".join([lines[i] for i in values])


def get_llm_response(documents: list[str], concept: Concept):
    supporting_documents = '\n\n\n'.join([document_filter(doc, concept.preferred_label.split(' ')) for doc in documents])
    prompt = f"""
    You are a domain expert in climate policy and terminology. You need to write an accurate definition for the concept provided below. You will also be given supporting documents. 
    Use XML tags for your output. 

    # Concept:

    {concept.to_markdown()}

    # Supporting Documents:
    {supporting_documents}

    # Your definition without any other context:
    """

    assert isinstance(prompt, str)

    response = llm.messages.create(
        model=model,
        messages=[{"role": "user", "content": str(prompt)}],
        max_tokens=1000,
    )

    return response



if __name__ == "__main__":
    correctable_concepts = get_correctable_concepts()

    for c, statement_to_references in list(correctable_concepts.items())[:1]:
        all_references = list(chain(*statement_to_references.values()))

        all_parsed_reference_links = [f"https://r.jina.ai/{v}" for v in all_references]
        all_reference_contents = [str(get(link).content) for link in all_parsed_reference_links]

        response = get_llm_response(all_reference_contents, c)
        print(response)
