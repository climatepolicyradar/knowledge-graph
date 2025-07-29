import asyncio
import json
import random
from abc import ABC, abstractmethod
from typing import Annotated, Optional

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from typing_extensions import Self

from src.classifier.classifier import Classifier, ZeroShotClassifier
from src.classifier.uncertainty_mixin import UncertaintyMixin
from src.concept import Concept
from src.identifiers import Identifier
from src.labelled_passage import LabelledPassage
from src.span import Span


class LLMResponse(BaseModel):
    """The response from the LLM"""

    marked_up_text: str = Field(
        description="The input text, reproduced exactly as it was given, with supplemental </ concept> tags where appropriate"
    )
    reasoning: str = Field(
        description="Justification for why the concept was identified in the supplied text, or why not"
    )


class LLMOutputMismatchError(Exception):
    """Raised when the LLM output text does not match the input text after removing tags."""

    def __init__(self, input_text: str, output_text: str):
        super().__init__(
            "Output text does not match input text.\n"
            f"Input:\t{input_text}\n"
            f"Output:\t{output_text}\n"
        )


DEFAULT_SYSTEM_PROMPT = """
You are a specialist climate policy analyst, tasked with identifying mentions of 
concepts in climate policy documents. You will mark up references to concepts with 
XML tags.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Instructions:

1. Read through each passage carefully, thinking about the concept.
2. Identify any mentions of the concept, including direct references and related terms.
3. Surround each identified mention with <concept> tags.
4. If a passage contains multiple instances, each one should be tagged separately.
5. If a passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
6. If an entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag.
7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
"""


class BaseLLMClassifier(Classifier, ZeroShotClassifier, UncertaintyMixin, ABC):
    """A classifier that uses an LLM to predict the presence of a concept in a text."""

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
    ):
        super().__init__(concept)
        self.concept = concept
        self.model_name = model_name
        self.system_prompt_template = system_prompt_template
        self.random_seed = random_seed

        assert "{concept_description}" in system_prompt_template, (
            "System prompt must contain {concept_description}"
        )

        self.system_prompt = system_prompt_template.format(
            concept_description=self.concept.to_markdown()
        )

        self.agent = self._create_agent()

    @abstractmethod
    def _create_agent(self) -> Agent:
        """Create the pydantic-ai agent for the classifier."""
        raise NotImplementedError

    @property
    def id(self) -> Identifier:
        """Return a deterministic, human-readable identifier for the classifier."""
        return Identifier.generate(
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

    def _validate_response(self, input_text: str, response: str) -> None:
        """Make sure the output text does not augment the input text in unexpected ways"""
        input_sanitised = LabelledPassage.sanitise(input_text)
        output_sanitised = LabelledPassage.sanitise(
            # remove the concept tags from the LLM output
            response.replace("<concept>", "").replace("</concept>", "")
        )
        if input_sanitised != output_sanitised:
            raise LLMOutputMismatchError(
                input_text=input_sanitised, output_text=output_sanitised
            )

    def get_variant_sub_classifier(self) -> Self:
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

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        response: AgentRunResult[LLMResponse] = self.agent.run_sync(  # type: ignore[assignment]
            text,
            model_settings=ModelSettings(seed=self.random_seed or 42),  # type: ignore[arg-type]
        )
        self._validate_response(input_text=text, response=response.data.marked_up_text)
        return Span.from_xml(
            xml=response.data.marked_up_text,
            concept_id=self.concept.wikibase_id,
            labellers=[str(self)],
        )

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """Predict whether the supplied texts contain instances of the concept."""

        async def run_predictions():
            async_responses = [
                self.agent.run(
                    text,
                    model_settings=ModelSettings(seed=self.random_seed or 42),  # type: ignore[arg-type]
                )
                for text in texts
            ]
            return await asyncio.gather(*async_responses)

        # Create a single event loop for all batches
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Run all predictions in the batch in parallel
            responses: list[AgentRunResult[LLMResponse]] = loop.run_until_complete(  # type: ignore[assignment]
                run_predictions()
            )
        finally:
            # Only close the loop if we created it
            if loop is not asyncio.get_event_loop():
                loop.close()

        batch_spans: list[list[Span]] = []

        for text, response in zip(texts, responses):
            try:
                self._validate_response(
                    input_text=text, response=response.data.marked_up_text
                )

                spans = Span.from_xml(
                    xml=response.data.marked_up_text,
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                )

                batch_spans.append(
                    [
                        # Use the original input texts for the output spans
                        span.model_copy(update={"text": text}, deep=True)
                        for span in spans
                    ]
                )
            except LLMOutputMismatchError:
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
        ] = "gemini-2.0-flash",
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
            self.model_name,
            system_prompt=self.system_prompt,
            result_type=LLMResponse,
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
        ollama_model = OpenAIModel(
            model_name=self.model_name,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
        return Agent(  # type: ignore[return-value]
            model=ollama_model,
            system_prompt=self.system_prompt,
            result_type=LLMResponse,
        )
