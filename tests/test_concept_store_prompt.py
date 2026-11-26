import pytest

from knowledge_graph.classifier.concept_store_prompt import (
    build_system_prompt_template,
    labelling_principles_version,
)
from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession


def test_build_system_prompt_template__keeps_braces_and_fills_concept():
    template = build_system_prompt_template(["Label {braces} literally."])
    prompt = LLMClassifierPrompt(system_prompt_template=template)
    concept = Concept(wikibase_id=WikibaseID("Q8"), preferred_label="greenhouse gas")

    formatted = prompt.format(concept)

    assert "Label {braces} literally." in formatted
    assert "greenhouse gas" in formatted
    assert "<labelling_principles>" in formatted


def test_parsed_html_to_prompt_text__strips_provenance_lines():
    data = {
        "parse": {
            "text": {"*": "<h3>Principle</h3><p>Since 2024-08-01</p><p>Label it.</p>"}
        }
    }

    text = WikibaseSession._parsed_html_to_prompt_text(data, "Help:Test")

    assert "Since 2024" not in text
    assert "Label it." in text


def test_parsed_html_to_prompt_text__raises_for_missing_page():
    data = {"error": {"code": "missingtitle", "info": "The page doesn't exist."}}

    with pytest.raises(ValueError, match="Help:Missing"):
        WikibaseSession._parsed_html_to_prompt_text(data, "Help:Missing")


def test_labelling_principles_version__is_stable_and_changes_with_text():
    version = labelling_principles_version("Label it.")

    assert version == labelling_principles_version("Label it.")
    assert version != labelling_principles_version("Label it!")
    assert len(version) == 12
