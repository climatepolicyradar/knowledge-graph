import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Optional, Sequence

import httpx
import nest_asyncio
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from typing_extensions import Self

from knowledge_graph.classifier.classifier import (
    Classifier,
    VariantEnabledClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span, SpanXMLConceptFormattingError
from knowledge_graph.wikibase import WikibaseSession

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """The response from the LLM"""

    marked_up_text: str = Field(
        description="The input text, reproduced exactly as it was given, with supplemental </ concept> tags where appropriate"
    )
    reasoning: str = Field(
        description="Justification for why the concept was identified in the supplied text, or why not"
    )


DEFAULT_SYSTEM_PROMPT = """
You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
These documents are mostly drawn from a climate and development context.
You will mark up references to concepts with XML tags.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Instructions:

1. Read through each passage carefully, thinking about the concept and different ways it can be used in documents.
2. Identify any mentions of the concept, including references that are not included as an example, but which match the definition.
3. Surround each identified mention with <concept> tags.
4. If a passage contains multiple instances, each one should be tagged separately.
5. If a passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
6. If an entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag. Skip this step if you have tagged any concept mentions so far.
7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
8. Double check that you have tagged all mentions of the concept and that every tagged part is describing an actual mention of that concept.
"""


@dataclass
class ConceptSettings:
    """Settings for determining how a concept is used in an LLMClassifier"""

    include_direct_subconcepts: bool = Field(
        description="Whether to include direct subconcepts and their definitions in the prompt"
    )
    labelling_guidelines: Optional[str] = Field(
        description="An optional set of labelling guidelines. Will be prefixed by the phrase 'labelling guidelines for this concept' in the system prompt"
    )

    @property
    def requires_wikibase(self) -> bool:
        """Whether the settings selected require Wikibase to be passed to the classifier"""

        fields_requiring_wikibase = [self.include_direct_subconcepts]

        return any(fields_requiring_wikibase)


class BaseLLMClassifier(Classifier, ZeroShotClassifier, VariantEnabledClassifier, ABC):
    """
    A classifier that uses an LLM to predict the presence of a concept in a text.

    Does not output prediction probabilities, so spans identified by this classifier
    will not have prediction_probability values set.
    """

    def __init__(
        self,
        concept: Concept,
        model_name: Annotated[
            str,
            Field(
                description=("The name of the model to use"),
            ),
        ],
        system_prompt_template: Annotated[
            str,
            Field(
                description=(
                    "The unformatted system prompt for the LLM, with values in {}. "
                    "Should be able to be populated with {concept_description} and "
                    "{examples} parameters."
                ),
            ),
        ] = DEFAULT_SYSTEM_PROMPT,
        random_seed: Annotated[
            Optional[int],
            Field(
                description=(
                    "Random seed for the classifier. Used for reproducibility and the "
                    "ability to spawn multiple distinguishable classifiers at the same "
                    "time"
                )
            ),
        ] = 42,
        concept_settings: Annotated[
            Optional[ConceptSettings],
            Field(
                description="Optional settings for how the concept is used in the system prompt."
            ),
        ] = None,
        wikibase: Annotated[
            Optional[WikibaseSession],
            Field(
                description="Optional Wikibase session. Might be required if concept_settings is set.",
            ),
        ] = None,
    ):
        super().__init__(concept)
        self.concept = concept
        self.model_name = model_name
        self.system_prompt_template = system_prompt_template
        self.random_seed = random_seed

        self.system_prompt = self.create_system_prompt(
            system_prompt_template=system_prompt_template,
            concept_settings=concept_settings,
            wikibase=wikibase,
        )

        self.agent = self._create_agent()

    def create_system_prompt(
        self,
        system_prompt_template: str,
        concept_settings: Optional[ConceptSettings],
        wikibase: Optional[WikibaseSession],
    ) -> str:
        """
        Create a system prompt from the prompt template, and optional settings.

        The features of Concept.to_markdown() which rely on Wikibase (i.e. concept
        neighbourhood) are not used, even if wikibase is not None. This is to keep the
        behaviour consistent.
        """

        if "{concept_description}" not in system_prompt_template:
            raise ValueError("System prompt must contain {concept_description}")

        if concept_settings is None:
            concept_description = self.concept.to_markdown(wikibase=None)

        elif concept_settings.requires_wikibase and wikibase is None:
            raise ValueError(
                "Wikibase session must be provided to LLMClassifier if ConceptSettings are set."
            )

        else:
            concept_description = self.concept.to_markdown()

            if concept_settings.include_direct_subconcepts:
                # Based on the logic in ConceptSettings and the elif statement above
                # this should always be true, so this is for the type checker.
                assert wikibase is not None

                # Re-fetch the concept, as it could've been provided with recursive
                # subconcepts
                concept_non_recursive = wikibase.get_concept(
                    self.concept.wikibase_id,
                    include_labels_from_subconcepts=False,
                    include_recursive_subconcept_of=False,
                    include_recursive_has_subconcept=False,
                )

                direct_subconcepts = wikibase.get_concepts(
                    concept_non_recursive.has_subconcept
                )
                _n = "\n"
                direct_subconcepts_string = _n.join(
                    [
                        f" - {subconcept.preferred_label}: {subconcept.definition}"
                        for subconcept in direct_subconcepts
                    ]
                )

                direct_subconcepts_description = f"""## Direct subconcepts
                
                The concept has the following direct subconcepts, which are semantically/conceptually part of the concept. Each subconcept is given by its name followed by its description.
                
                {direct_subconcepts_string}
                """

                concept_description = (
                    concept_description + "\n" + direct_subconcepts_description
                )

        return system_prompt_template.format(
            concept_description=self.concept.to_markdown(wikibase=None)
        )

    @abstractmethod
    def _create_agent(self) -> Agent:
        """Create the pydantic-ai agent for the classifier."""
        raise NotImplementedError

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.model_name,
            self.system_prompt,
            self.random_seed,
        )

    def __hash__(self) -> int:
        """Overrides the default hash function, to enrich the hash with metadata"""
        return hash(self.id)

    def __repr__(self):
        """Return a string representation of the classifier."""
        values: dict[str, str | int | None] = {
            "model_name": self.model_name,
            "id": self.id,
        }
        if self.random_seed:
            values["random_seed"] = self.random_seed
        values_string = json.dumps(values)[1:-1].replace(": ", "=")
        return f'{self.name}("{self.concept.preferred_label}", {values_string})'

    def get_variant(self) -> Self:
        """Get a variant of the classifier, using a different random seed."""
        return type(self)(
            concept=self.concept,
            model_name=self.model_name,
            system_prompt_template=self.system_prompt_template,
            random_seed=random.randint(0, 1000000),
        )

    def __getstate__(self):
        """Handle pickling by removing the unpickleable agent instance."""
        state = self.__dict__.copy()
        # Remove the agent instance as it contains unpickleable components
        del state["agent"]
        return state

    def __setstate__(self, state):
        """Recreate the agent instance when unpickling."""
        self.__dict__.update(state)
        self.agent = self._create_agent()

    @staticmethod
    def _check_and_nest_event_loop():
        """
        Use nest_asyncio to be able to run nested event loops.

        This is needed for the predict methods if they are running within async outer
        loops, as the pydantic-ai library uses async calls to LLM APIs.
        """

        # Check whether an event loop is already running. Python can't run nested
        # async processes, so if there's a running loop then we use `nest_asyncio` to
        # make the prediction be able to run.
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                nest_asyncio.apply()
        except RuntimeError:
            pass

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :param float | None threshold: Optional prediction threshold
        :return list[Span]: A list of spans identified in the text
        """
        if threshold is not None:
            logger.warning(
                f"`threshold` parameter ignored - {self.__class__.__name__} does not output prediction probabilities",
            )

        self._check_and_nest_event_loop()

        try:
            response: AgentRunResult[LLMResponse] = self.agent.run_sync(  # type: ignore[assignment]
                text,
                model_settings=ModelSettings(seed=self.random_seed or 42),  # type: ignore[arg-type]
            )
        except UnexpectedModelBehavior as e:
            logger.warning(
                f"LLM failed to produce valid response after retries: {e}. "
                f"Text (truncated): {text[:100]}..."
            )
            return []

        try:
            return Span.from_xml(
                xml=response.output.marked_up_text,
                concept_id=self.concept.wikibase_id,
                labellers=[str(self)],
                input_text=text,
            )
        except (SpanXMLConceptFormattingError, ValidationError) as e:
            logger.warning(f"Prediction failed: {e}")
            return []

        except Exception as e:
            logger.warning(f"Prediction failed with unexpected exception type. {e}")
            return []

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param Sequence[str] texts: The texts to predict on
        :param float | None threshold: Optional prediction threshold
        :return list[list[Span]]: A list of span lists identified in each text
        """
        self._check_and_nest_event_loop()

        async def run_predictions():
            async_responses = [
                self.agent.run(
                    text,
                    model_settings=ModelSettings(seed=self.random_seed or 42),  # type: ignore[arg-type]
                )
                for text in texts
            ]
            return await asyncio.gather(*async_responses, return_exceptions=True)

        # Create a single event loop for all batches
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Run all predictions in the batch in parallel
            responses: list[AgentRunResult[LLMResponse] | Exception] = (
                loop.run_until_complete(  # type: ignore[assignment]
                    run_predictions()
                )
            )
        finally:
            # Only close the loop if we created it
            if loop is not asyncio.get_event_loop():
                loop.close()

        batch_spans: list[list[Span]] = []

        for text, response in zip(texts, responses):
            # Handle exceptions that occurred during async execution
            if isinstance(response, Exception):
                if isinstance(response, UnexpectedModelBehavior):
                    logger.warning(
                        f"LLM failed to produce valid response after retries: {response}. "
                        f"Text (truncated): {text[:100]}..."
                    )
                else:
                    logger.warning(
                        f"Prediction failed with exception: {response}. "
                        f"Text (truncated): {text[:100]}..."
                    )
                batch_spans.append([])
                continue

            try:
                spans = Span.from_xml(
                    xml=response.output.marked_up_text,
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                    input_text=text,
                )
                batch_spans.append(
                    [
                        # Use the original input texts for the output spans
                        span.model_copy(update={"text": text}, deep=True)
                        for span in spans
                    ]
                )

            except (SpanXMLConceptFormattingError, ValidationError) as e:
                logger.warning(f"Prediction failed: {e}")
                batch_spans.append([])

            except Exception as e:
                logger.warning(f"Prediction failed with unexpected exception type. {e}")
                batch_spans.append([])

        return batch_spans


class LLMClassifier(BaseLLMClassifier):
    """A classifier which uses a third-party LLM to identify concept mentions in text"""

    def __init__(
        self,
        concept: Concept,
        model_name: Annotated[
            str,
            Field(
                description=(
                    "The name of the model to use. See https://ai.pydantic.dev/models/ "
                    "for a list of available models and the necessary environment "
                    "variables needed to run each."
                ),
            ),
        ] = "openrouter:openai/gpt-5",
        system_prompt_template: Annotated[
            str,
            Field(
                description=(
                    "The unformatted system prompt for the LLM, with values in {}. "
                    "Should be able to be populated with {concept_description} and "
                    "{examples} parameters."
                ),
            ),
        ] = DEFAULT_SYSTEM_PROMPT,
        random_seed: Annotated[
            int,
            Field(
                description=(
                    "Random seed for the classifier. "
                    "Used for reproducibility and the ability to spawn multiple "
                    "distinguishable classifiers at the same time"
                )
            ),
        ] = 42,
    ):
        super().__init__(
            concept=concept,
            model_name=model_name,
            system_prompt_template=system_prompt_template,
            random_seed=random_seed,
        )

    def _create_agent(self) -> Agent:  # type: ignore[type-arg]
        return Agent(  # type: ignore[return-value]
            model=self.model_name,
            system_prompt=self.system_prompt,
            output_type=LLMResponse,
        )


class LocalLLMClassifier(BaseLLMClassifier):
    """
    A classifier which uses a local LLM served to identify concept mentions in text.

    This classifier interacts with a local Ollama instance, which must be running
    with the specified model downloaded. See https://ollama.com/ for more details.
    """

    def __init__(
        self,
        concept: Concept,
        model_name: Annotated[
            str,
            Field(
                description=(
                    "The name of the model to use. "
                    "See https://ollama.com/library for a list of options."
                ),
            ),
        ] = "gemma3n:e4b",
        system_prompt_template: Annotated[
            str,
            Field(
                description=(
                    "The unformatted system prompt for the LLM, with values in {}. "
                    "Should be able to be populated with {concept_description} and "
                    "{examples} parameters."
                ),
            ),
        ] = DEFAULT_SYSTEM_PROMPT,
        random_seed: Annotated[
            int,
            Field(
                description=(
                    "Random seed for the classifier. "
                    "Used for reproducibility and the ability to spawn multiple "
                    "distinguishable classifiers at the same time"
                )
            ),
        ] = 42,
    ):
        try:
            httpx.get("http://localhost:11434", timeout=1)
        except httpx.ConnectError as e:
            raise ConnectionError(
                "Ollama isn't running! Make sure you've downloaded and are running "
                "Ollama before trying to use a local LLM classifier. "
                "See https://ollama.com/ for more details."
            ) from e
        super().__init__(
            concept=concept,
            model_name=model_name,
            system_prompt_template=system_prompt_template,
            random_seed=random_seed,
        )

    def _create_agent(self) -> Agent:
        ollama_model = OpenAIChatModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
        return Agent(  # type: ignore[return-value]
            model=ollama_model,
            system_prompt=self.system_prompt,
            output_type=LLMResponse,
        )
