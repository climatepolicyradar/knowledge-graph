import re

from pydantic import BaseModel
from pydantic_ai import Agent

from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID

# Separators that indicate the model has packed several keywords into one string.
_RUNON_SPLIT_RE = re.compile(r"[,\n;|/]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Real alternative labels are short; longer strings are run-on/descriptive noise.
_MAX_KEYWORD_WORDS = 8


def sanitise_keywords(keywords: list[str]) -> list[str]:
    """
    Clean LLM-generated keywords before scoring or matching.

    Weaker models append non-Latin characters (e.g. CJK), emoji, run-on
    concatenations and natural-language descriptions, all of which are pure
    false-positive mass for an exact-string-matching classifier. This:

    - splits run-on strings on commas/semicolons/newlines/pipes/slashes,
    - drops any keyword containing non-ASCII characters (emoji, CJK, ...),
    - drops over-long phrases that read as descriptions rather than terms,
    - collapses whitespace and de-duplicates case-insensitively.

    :param list[str] keywords: Raw keywords from the LLM
    :return list[str]: Cleaned keywords, order preserved
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        for piece in _RUNON_SPLIT_RE.split(raw):
            keyword = _WHITESPACE_RE.sub(" ", piece).strip()
            if not keyword or not keyword.isascii():
                continue
            if len(keyword.split()) > _MAX_KEYWORD_WORDS:
                continue
            key = keyword.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(keyword)
    return cleaned


class ExpandedKeywords(BaseModel):
    """Structured output from the keyword expansion LLM."""

    positive_keywords: list[str]
    negative_keywords: list[str]


DEFAULT_KEYWORD_EXPANSION_PROMPT = """You are a domain expert in climate policy and terminology. Your task is to generate alternative labels (aliases) for a concept, following the conventions used in a curated concept store.

First, carefully review the following description of the concept of "{PREFERRED_LABEL}":

<concept_description>
{CONCEPT_DESCRIPTION}
</concept_description>

An alternative label is a term that means the SAME THING as the concept - a word a reader could substitute for the preferred label and have it still refer to exactly this concept. It is NOT a related, broader, narrower, or example term.

INCLUDE as positive_keywords:
1. Direct synonyms (e.g. "youth" for "child")
2. Plural and singular forms (e.g. "greenhouse gases" for "greenhouse gas")
3. Split-compound / extra-space variants (e.g. "green house gas" for "greenhouse gas")
4. Scientific or common-name equivalents (e.g. "influenza" for "flu")
5. Genuinely interchangeable acronyms and abbreviations (e.g. "CO2" for "carbon dioxide")
6. Gendered forms where the preferred label is gender-neutral (e.g. "fireman" for "firefighter")

Do NOT INCLUDE as positive_keywords:
- Related, broader, narrower, or sub-category terms, or causes/consequences/examples (these belong to OTHER concepts, not this one - e.g. for "open burning" do not add "biomass burning" or "agricultural burning")
- Descriptive paraphrases or multi-word phrases nobody writes verbatim (e.g. "accelerated ice motion")
- Misspellings or typos (e.g. "carbon dioxyde")
- Case-only variants (e.g. both "GHG" and "ghg") or punctuation-only variants (e.g. both "capacity building" and "capacity-building") - these are matched automatically, so list only one form

Separately, generate negative_keywords: specific phrases that CONTAIN the concept's words but refer to something else, and should be excluded to avoid false positives (e.g. for "gas", the negative keyword "greenhouse gas").

Format your response as valid JSON with two lists:
{{
    "positive_keywords": ["term1", "term2", ...],
    "negative_keywords": ["exclude1", "exclude2", ...]
}}

Be precise and conservative: only include a term if you are confident it is a true synonym or surface variant of the concept. Do not pad the list - a shorter list of exact synonyms is better than a long list containing related terms.
"""


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
        model: str = "openrouter:google/gemini-3-flash-preview",
        prompt_template: str | None = None,
    ):
        self.original_concept = concept
        self.concept = concept
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_KEYWORD_EXPANSION_PROMPT
        self.agent: Agent[None, ExpandedKeywords] = Agent(
            model=self.model, output_type=ExpandedKeywords
        )

        # Initialize with original concept so that we can always fall back to it
        super().__init__(self.concept)

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(self.name, self.concept.id)

    def _generate_prompt(self, preferred_label: str, concept_description: str) -> str:
        """Generate the prompt for keyword expansion from the given concept view."""

        prompt = "\n".join(line.strip() for line in self.prompt_template.split("\n"))

        prompt = prompt.format(
            PREFERRED_LABEL=preferred_label,
            CONCEPT_DESCRIPTION=concept_description,
        )

        return prompt

    def _generate_keywords(
        self, preferred_label: str, concept_description: str
    ) -> dict[str, list[str]]:
        """
        Call the LLM to expand keywords for the given concept view.

        This is the raw generation step: it does *not* merge in the concept's own
        labels, so callers (e.g. evaluation harnesses) can compare the generated
        keywords against held-out ground truth without leakage.

        :param str preferred_label: The concept's preferred label
        :param str concept_description: A (possibly masked) markdown view of the concept
        :return dict[str, list[str]]: ``{"positive_keywords": [...], "negative_keywords": [...]}``.
            Returns empty lists if the LLM call fails.
        """
        prompt = self._generate_prompt(preferred_label, concept_description)
        try:
            expanded = self.agent.run_sync(prompt).output
        except Exception as e:  # noqa: BLE001 - fall back to no expansion on any failure
            print(f"Warning: keyword expansion LLM call failed: {e}")
            return {"positive_keywords": [], "negative_keywords": []}

        return {
            "positive_keywords": sanitise_keywords(expanded.positive_keywords),
            "negative_keywords": sanitise_keywords(expanded.negative_keywords),
        }

    def fit(self, **kwargs) -> "KeywordExpansionClassifier":
        """
        Expand the keyword lists using an LLM and train the classifier.

        :return KeywordExpansionClassifier: The trained classifier
        """
        expanded_keywords = self._generate_keywords(
            preferred_label=self.original_concept.preferred_label,
            concept_description=self.original_concept.to_markdown(),
        )

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
