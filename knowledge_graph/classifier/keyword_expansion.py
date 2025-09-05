import json
import os

from anthropic import Anthropic

from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID


class KeywordExpansionClassifier(KeywordClassifier):
    """
    A classifier that uses an LLM to expand the set of keywords used for matching.

    This classifier takes the initial concept keywords and uses an LLM to generate
    additional related terms. For example, if given the concept "horse", it might
    expand to include terms like "pony", "mare", "stallion", etc. It then uses these
    expanded terms for matching using the logic of the parent KeywordClassifier.
    """

    def __init__(
        self,
        concept: Concept,
        model: str = "claude-3-5-haiku-20241022",
    ):
        self.original_concept = concept
        self.concept = concept
        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Initialize with original concept so that we can always fall back to it
        super().__init__(self.concept)

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(self.name, self.concept.id)

    def _generate_prompt(self) -> str:
        """Generate the prompt for keyword expansion."""

        prompt_template = """You are a domain expert in climate policy and terminology. Your task is to expand a list of keywords related to a specific concept, while being mindful of any terms that should be explicitly excluded.

        First, carefully review the following description of the concept of "{PREFERRED_LABEL}":

        <concept_description>
        {CONCEPT_DESCRIPTION}
        </concept_description>

        Please generate additional keywords that are strongly related to this concept. Focus on:
        1. Common variations and synonyms
        2. Acronyms and abbreviations
        3. Technical or domain-specific terminology
        4. Related subcategories
        5. Common misspellings or alternative spellings
        6. Plural/singular forms if not already included

        Also generate additional negative keywords - terms that might be mistakenly matched but should be excluded.

        Format your response as valid JSON with two lists:
        {{
            "positive_keywords": ["term1", "term2", ...],
            "negative_keywords": ["exclude1", "exclude2", ...]
        }}

        Be precise and avoid overly broad terms that might lead to false positives. Each term should have a strong, direct relationship to the core concept.
        """

        prompt = "\n".join(line.strip() for line in prompt_template.split("\n"))

        prompt = prompt.format(
            PREFERRED_LABEL=self.original_concept.preferred_label,
            CONCEPT_DESCRIPTION=self.original_concept.to_markdown(),
        )

        return prompt

    def fit(self, **kwargs) -> "KeywordExpansionClassifier":
        """
        Expand the keyword lists using an LLM and train the classifier.

        :return KeywordExpansionClassifier: The trained classifier
        """
        prompt = self._generate_prompt()
        response = (
            self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            .content[0]
            .text  # pyright: ignore[reportAttributeAccessIssue]
        )

        try:
            expanded_keywords = json.loads(response)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse LLM response for keyword expansion: {e}")
            # Fall back to original concept if expansion fails
            super().__init__(self.original_concept)
            return self

        # Create a new concept with the expanded set of keywords
        positive_keywords = list(
            set(expanded_keywords["positive_keywords"] + self.concept.all_labels)
        )
        negative_keywords = list(
            set(expanded_keywords["negative_keywords"] + self.concept.negative_labels)
        )
        expanded_concept = self.concept.model_copy(
            update={
                "alternative_labels": positive_keywords,
                "negative_labels": negative_keywords,
            }
        )

        # Reinitialize parent with expanded concept
        self.concept = expanded_concept
        super().__init__(expanded_concept)

        return self
