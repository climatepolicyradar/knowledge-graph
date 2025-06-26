import asyncio
from asyncio import gather
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.settings import ModelSettings

from src.classifier.classifier import Classifier, ZeroShotClassifier
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


class LLMClassifier(Classifier, ZeroShotClassifier):
    """A classifier that uses an LLM to predict the presence of a concept in a text."""

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
        super().__init__(concept)
        self.concept = concept
        self.model_name = model_name
        self.system_prompt_template = system_prompt_template

        assert "{concept_description}" in system_prompt_template, (
            "System prompt must contain {concept_description}"
        )

        self.system_prompt = system_prompt_template.format(
            concept_description=self.concept.to_markdown()
        )

        self.agent = Agent(
            self.model_name,
            system_prompt=self.system_prompt,
            result_type=LLMResponse,
        )

        self.random_seed = random_seed

    @property
    def id(self) -> Identifier:
        """Overrides the default hash function, to enrich the hash with metadata"""
        return Identifier.generate(
            self.name,
            self.concept.__hash__(),
            self.version if self.version else None,
            self.model_name,
            self.system_prompt_template,
            self.random_seed,
        )

    def __hash__(self) -> int:
        """Return a hash of the classifier."""
        return hash(self.id)

    def __repr__(self):
        """Return a string representation of the classifier."""
        return f'{self.name}({self.concept.preferred_label}, model_name="{self.model_name}", id={self.id})'

    def __getstate__(self):
        """Handle pickling by removing the unpickleable agent instance."""
        state = self.__dict__.copy()
        # Remove the agent instance as it contains unpickleable components
        del state["agent"]
        return state

    def __setstate__(self, state):
        """Recreate the agent instance when unpickling."""
        self.__dict__.update(state)
        # Recreate the agent instance
        self.agent = Agent(
            self.model_name,
            system_prompt=self.system_prompt,
            result_type=LLMResponse,
        )

    def _validate_response(self, input_text: str, response: LLMResponse) -> None:
        """Make sure the output text does not augment the input text in unexpected ways"""
        input_sanitised = LabelledPassage.sanitise(input_text)
        output_sanitised = LabelledPassage.sanitise(
            # remove the concept tags from the LLM output
            response.marked_up_text.replace("<concept>", "").replace("</concept>", "")
        )
        if input_sanitised != output_sanitised:
            raise LLMOutputMismatchError(
                input_text=input_sanitised, output_text=output_sanitised
            )

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        response: AgentRunResult[LLMResponse] = self.agent.run_sync(
            text, model_settings=ModelSettings(seed=self.random_seed)
        )
        self._validate_response(input_text=text, response=response.data)
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
                    text, model_settings=ModelSettings(seed=self.random_seed)
                )
                for text in texts
            ]
            return await gather(*async_responses)

        # Create a single event loop for all batches
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Run all predictions in the batch in parallel
            responses: list[AgentRunResult[LLMResponse]] = loop.run_until_complete(
                run_predictions()
            )
        finally:
            # Only close the loop if we created it
            if loop is not asyncio.get_event_loop():
                loop.close()

        batch_spans: list[list[Span]] = []

        for text, response in zip(texts, responses):
            try:
                self._validate_response(input_text=text, response=response.data)

                # str(self) includes the id now, as this is important in distinguishing between different instances
                # of the same classifier + config, when comparing spans with each other
                spans = Span.from_xml(
                    xml=response.data.marked_up_text,
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                )

                # We validate that the sanitised versions of input and output text are
                # identical, but that doesn't guarantee that the un-sanitised responses
                # match the input! For example, whitespace differences are deliberately
                # ignored by the sanitisation step.
                # To guard against the possibility of the LLM corrupting the input in
                # subtle ways that the sanitisation step misses, we replace the text
                # in the output spans with the original input text
                batch_spans.append(
                    [
                        span.model_copy(update={"text": text}, deep=True)
                        for span in spans
                    ]
                )
            except LLMOutputMismatchError:
                batch_spans.append([])

        return batch_spans
