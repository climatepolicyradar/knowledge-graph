import string
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from knowledge_graph.classifier.large_language_model import (
    DEFAULT_SYSTEM_PROMPT,
    LLMClassifierPrompt,
)
from knowledge_graph.identifiers import WikibaseID

DESCRIPTION_WIKIBASE_LENGTH_LIMIT = 2500
DEFINITION_WIKIBASE_LENGTH_LIMIT = 2500


class ConceptOverrides(BaseModel):
    """Definition/description here must EXCEED the length limit."""

    model_config = ConfigDict(extra="forbid")

    definition: str | None = None
    description: str | None = None
    alternative_labels: list[str] | None = None
    negative_labels: list[str] | None = None

    @field_validator("definition", "description")
    @classmethod
    def _must_exceed_store(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Reject a definition/description that fits in the concept store (does not exceed length limit)."""
        if v is None:
            return v
        cap = (
            DEFINITION_WIKIBASE_LENGTH_LIMIT
            if info.field_name == "definition"
            else DESCRIPTION_WIKIBASE_LENGTH_LIMIT
        )
        if len(v) <= cap:
            raise ValueError(
                f"{info.field_name} is {len(v)} chars (<= {cap}); it fits the concept store - "
                "put it in Wikibase instead of overriding here."
            )
        return v

    def as_overrides(self) -> dict[str, Any]:
        """Return the set override fields as a dict (drops unset/None)."""
        return self.model_dump(exclude_none=True)


class SamplingConfig(BaseModel):
    """Maps to run_sampling() params (sample.py)."""

    dataset_name: Literal["balanced", "combined"] = "balanced"
    sample_size: int = 130
    min_negative_proportion: float = 0.1
    max_negative_proportion: float | None = None
    corpus_types_include: list[str] | None = None
    corpus_types_exclude: list[str] | None = None
    max_size_to_sample_from: int = 500_000

    def to_run_sampling_kwargs(self) -> dict[str, Any]:
        """Exactly the run_sampling() sampling kwargs (drops None so run_sampling defaults win)."""
        return self.model_dump(exclude_none=True)


class LLMClassifierConfig(BaseModel):
    """Model + prompt ({slot}s filled from related_definitions)."""

    model_name: str
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT
    labelling_guidelines: str | None = None
    related_definitions: list[WikibaseID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _placeholders_match_related(self):
        """Every {slot} in labelling_guidelines must be declared in related_definitions, & vice versa."""
        slots = (
            {
                n
                for _, n, _, _ in string.Formatter().parse(self.labelling_guidelines)
                if n
            }
            if self.labelling_guidelines
            else set()
        )
        expected = {str(wid) for wid in self.related_definitions}
        if slots != expected:
            raise ValueError(
                f"labelling_guidelines slots {sorted(slots)} != "
                f"related_definitions {sorted(expected)}"
            )
        return self

    def _render_guidelines(self, definitions: dict[WikibaseID, str]) -> str | None:
        """Fill {slot} placeholders with the related concepts' definitions (passed by the caller)."""
        if self.labelling_guidelines is None:
            return None
        if not self.related_definitions:
            return self.labelling_guidelines
        slots = {str(wid): definitions[wid] for wid in self.related_definitions}
        return self.labelling_guidelines.format(**slots)

    def to_classifier_kwargs(
        self, definitions: dict[WikibaseID, str] | None = None
    ) -> dict[str, Any]:
        """Build the LLMClassifier kwargs (model + prompt with rendered guidelines)."""
        return {
            "model_name": self.model_name,
            "system_prompt_template": LLMClassifierPrompt(
                system_prompt_template=self.system_prompt_template,
                labelling_guidelines=self._render_guidelines(definitions or {}),
            ),
        }


class BERTClassifierConfig(BaseModel):
    """Maps to BertBasedClassifier kwargs + the train.py linkage to the LLM-labelled data."""

    model_name: str = "answerdotai/ModernBERT-base"
    unfreeze_layers: int = 0
    limit_training_samples: int | None = None
    training_data_wandb_path: str | None = None

    def to_classifier_kwargs(self) -> dict[str, Any]:
        """Build the BertBasedClassifier kwargs."""
        return {
            "model_name": self.model_name,
            "unfreeze_layers": self.unfreeze_layers,
        }


class CustomClassifierConfig(BaseModel):
    """Config for custom concept classifier (one YAML file per concept)."""

    model_config = ConfigDict(extra="forbid")

    wikibase_id: WikibaseID
    concept_overrides: ConceptOverrides = Field(default_factory=ConceptOverrides)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    llm: LLMClassifierConfig
    bert: BERTClassifierConfig = Field(default_factory=BERTClassifierConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CustomClassifierConfig":
        """Load and validate a config from a YAML file."""
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
