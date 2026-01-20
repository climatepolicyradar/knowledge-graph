import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Annotated, Optional, Sequence

import httpx
import nest_asyncio
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
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
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span, SpanXMLConceptFormattingError

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


# System prompt template for AutoLLMClassifier with placeholder for optimized instructions
AUTO_DEFAULT_SYSTEM_PROMPT = """
You are a specialist analyst, tasked with identifying mentions of concepts in policy documents.
These documents are mostly drawn from a climate and development context.
You will mark up references to concepts with XML tags.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Instructions:

{instructions}
"""


# Default instructions for AutoLLMClassifier (used if fit() not called or optimization fails)
DEFAULT_INSTRUCTIONS = """
1. Read through each passage carefully, thinking about the concept and different ways it can be used in documents.
2. Identify any mentions of the concept, including references that are not included as an example, but which match the definition.
3. Surround each identified mention with <concept> tags.
4. If a passage contains multiple instances, each one should be tagged separately.
5. If a passage does not contain any instances, it should be reproduced exactly as given, without any additional tags.
6. If an entire passage refers to the concept without specific mentions, the entire passage should be wrapped in a <concept> tag. Skip this step if you have tagged any concept mentions so far.
7. The input text must be reproduced exactly, down to the last character, only adding concept tags.
8. Double check that you have tagged all mentions of the concept and that every tagged part is describing an actual mention of that concept.
"""


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

        response: AgentRunResult[LLMResponse] = self.agent.run_sync(  # type: ignore[assignment]
            text,
            model_settings=ModelSettings(seed=self.random_seed or 42),  # type: ignore[arg-type]
        )

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
        """Predict whether the supplied texts contain instances of the concept."""

        self._check_and_nest_event_loop()

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


class AutoLLMClassifier(BaseLLMClassifier):
    """
    An LLM classifier that uses DSPy to optimize labelling guidelines.

    This classifier automatically tunes the system prompt's instruction section
    to maximize passage-level F-beta score (F0.5 by default) on a binary classification
    task with precision focus. Unlike traditional LLM classifiers, it performs a
    supervised learning step (fit()) that uses MIPRO optimization to improve
    instructions based on labelled examples.

    The optimization process:

    1. Converts concept.labelled_passages to DSPy examples
    2. Uses MIPRO to optimize the instruction section of the prompt
    3. Uses existing count_passage_level_metrics() for F-beta evaluation (F0.5 by default)
    4. Stores optimized instructions in system_prompt_template
    5. Uses the optimized prompt for all subsequent predictions

    Model Configuration
    -------------------
    Three separate models can be configured for different phases:

    - classifier_model_name: Used for final inference after optimization
    - evaluation_model_name: Used for scoring during optimization (many calls, use cheap model)
    - proposal_model_name: Used for generating instruction candidates (use smart model)

    Example
    -------
    >>> classifier = AutoLLMClassifier(
    ...     concept=concept,
    ...     classifier_model_name="gpt-4o",           # Production inference
    ...     evaluation_model_name="gpt-4o-mini",      # Cheap model for scoring
    ...     proposal_model_name="gpt-4o",             # Smart model for instructions
    ... )

    Parameters
    ----------
    concept : Concept
        The concept to classify
    classifier_model_name : str, default="gemini-2.0-flash"
        Model for final inference after optimization
    system_prompt_template : str, default=AUTO_DEFAULT_SYSTEM_PROMPT
        Base system prompt with {concept_description} and {instructions} placeholders
    random_seed : int, default=42
        Random seed for reproducibility
    evaluation_model_name : str | None, default=None
        Model for evaluation during optimization. Defaults to classifier_model_name.
        Use a cheaper/faster model (e.g., gpt-4o-mini) since evaluation requires many calls.
    proposal_model_name : str | None, default=None
        Model for generating instruction candidates. Defaults to evaluation_model_name.
        Use a smarter model (e.g., gpt-4o, claude-3.5-sonnet) for better instructions.
    """

    def __init__(
        self,
        concept: Concept,
        classifier_model_name: Annotated[
            str,
            Field(
                description=(
                    "The model to use for final inference after optimization. "
                    "See https://ai.pydantic.dev/models/ for available models."
                ),
            ),
        ] = "gemini-2.0-flash",
        system_prompt_template: Annotated[
            str,
            Field(
                description=(
                    "The unformatted system prompt for the LLM, with {concept_description} "
                    "and {instructions} placeholders. Instructions will be optimized by DSPy."
                ),
            ),
        ] = AUTO_DEFAULT_SYSTEM_PROMPT,
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
        evaluation_model_name: Annotated[
            str | None,
            Field(
                description=(
                    "Model for evaluation during DSPy optimization. If None, uses "
                    "classifier_model_name. Use a cheaper model (e.g., gpt-4o-mini) "
                    "since evaluation requires many LLM calls."
                )
            ),
        ] = None,
        proposal_model_name: Annotated[
            str | None,
            Field(
                description=(
                    "Model for generating instruction candidates during optimization. "
                    "If None, uses evaluation_model_name. Use a smarter model "
                    "(e.g., gpt-4o, claude-3.5-sonnet) for higher quality instructions."
                )
            ),
        ] = None,
    ):
        # Store optimization settings and initialize instructions before calling super().__init__
        self.evaluation_model_name = evaluation_model_name or classifier_model_name
        self.proposal_model_name = proposal_model_name or self.evaluation_model_name
        self.optimized_instructions: str | None = None

        # Initialize with default instructions to satisfy template formatting
        formatted_template = system_prompt_template.format(
            concept_description="{concept_description}",
            instructions=DEFAULT_INSTRUCTIONS,
        )

        super().__init__(
            concept=concept,
            model_name=classifier_model_name,
            system_prompt_template=formatted_template,
            random_seed=random_seed,
        )

    @property
    def id(self) -> ClassifierID:
        """
        Return deterministic identifier for the classifier.

        Includes optimized_instructions in ID generation to ensure different
        optimized versions have different IDs.

        Returns
        -------
        ClassifierID
            Unique identifier for this classifier configuration
        """
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.model_name,
            self.system_prompt,
            self.random_seed,
            self.optimized_instructions or "",  # Changes ID after fit()
        )

    def _create_agent(self) -> Agent:  # type: ignore[type-arg]
        """
        Create pydantic-ai agent for inference using OpenRouter.

        Returns
        -------
        Agent
            Configured pydantic-ai agent
        """
        # Route through OpenRouter - requires OPENROUTER_API_KEY environment variable
        model_name = self.model_name
        if not model_name.startswith("openrouter:"):
            model_name = f"openrouter:{model_name}"

        return Agent(  # type: ignore[return-value]
            model=model_name,
            system_prompt=self.system_prompt,
            output_type=LLMResponse,
        )

    def _prepare_dspy_examples(
        self, passages: list[LabelledPassage], validation_size: float
    ) -> tuple[list, list]:
        """
        Convert LabelledPassage objects to DSPy Examples with stratified split.

        Each example has:

        - passage_text (input)
        - passage_id (for matching predictions to ground truth)
        - gold_spans (for F1 computation)

        Parameters
        ----------
        passages : list[LabelledPassage]
            Labelled passages from concept
        validation_size : float
            Proportion for validation set (e.g., 0.2 for 80/20 split)

        Returns
        -------
        tuple[list, list]
            (train_examples, val_examples) as DSPy Example objects
        """
        import dspy
        from sklearn.model_selection import train_test_split

        # Convert to DSPy format
        examples = []
        for passage in passages:
            example = dspy.Example(
                passage_text=passage.text,
                passage_id=passage.id,
                gold_spans=passage.spans,
            ).with_inputs("passage_text")

            examples.append(example)

        # Stratified split based on whether passage has concept spans
        labels = [
            any(span.concept_id == self.concept.wikibase_id for span in ex.gold_spans)
            for ex in examples
        ]

        train_examples, val_examples = train_test_split(
            examples,
            test_size=validation_size,
            random_state=self.random_seed,
            stratify=labels,
        )

        # Log split information
        train_positive = sum(
            any(span.concept_id == self.concept.wikibase_id for span in ex.gold_spans)
            for ex in train_examples
        )
        val_positive = sum(
            any(span.concept_id == self.concept.wikibase_id for span in ex.gold_spans)
            for ex in val_examples
        )

        logger.info(
            f"Split data: {len(train_examples)} train, {len(val_examples)} val\n"
            f"Train positive: {train_positive}/{len(train_examples)}\n"
            f"Val positive: {val_positive}/{len(val_examples)}"
        )

        return train_examples, val_examples

    def _create_dspy_lm(self, model_name: str | None = None, temperature: float = 0.7):
        """
        Create DSPy language model for optimization.

        Maps model names to appropriate DSPy LM classes using Litellm format.

        Parameters
        ----------
        model_name : str | None, default=None
            Model name to use. If None, uses self.evaluation_model_name.
        temperature : float, default=0.7
            Temperature for LLM sampling. Higher values (e.g., 0.7-1.0) produce
            more varied outputs, which is important for optimization to evaluate
            how different instructions affect model behavior. Set to 0 for
            deterministic outputs.

        Returns
        -------
        dspy.LM
            Configured DSPy language model
        """
        import dspy

        model_name = model_name or self.evaluation_model_name

        # DSPy uses Litellm under the hood, route all models through OpenRouter
        # Requires OPENROUTER_API_KEY environment variable
        # Temperature > 0 is critical for optimization - without it, all instruction
        # candidates may produce identical outputs, preventing meaningful comparison
        if model_name.startswith("openrouter/"):
            # Already has openrouter prefix
            openrouter_model = model_name
        else:
            # Prepend openrouter/ to model name
            openrouter_model = f"openrouter/{model_name}"

        # Reasoning models (o1, o3, gpt-5, etc.) require special parameters
        import re

        reasoning_pattern = re.compile(r"(o1|o3|gpt-5)", re.IGNORECASE)
        if reasoning_pattern.search(model_name):
            # Reasoning models require temperature=1.0 and max_tokens >= 16000
            return dspy.LM(model=openrouter_model, max_tokens=16000, temperature=1.0)

        return dspy.LM(model=openrouter_model, max_tokens=4000, temperature=temperature)

    def _extract_optimized_instructions(self, optimized_module) -> str:
        """
        Extract optimized instruction string from DSPy module.

        MIPRO stores optimized instructions in the signature's instruction field.

        Parameters
        ----------
        optimized_module : dspy.Module
            Optimized DSPy module from MIPRO

        Returns
        -------
        str
            Optimized instructions as plain text
        """
        try:
            # MIPRO stores optimized prompt in the predict module's signature
            signature = optimized_module.predict.signature

            # Extract the instructions field (MIPRO modifies the signature docstring)
            instructions = None
            if hasattr(signature, "instructions"):
                instructions = signature.instructions

            # Validate we got meaningful instructions
            if instructions and isinstance(instructions, str) and instructions.strip():
                logger.debug(
                    f"Extracted optimized instructions ({len(instructions)} chars)"
                )
                return instructions

            # Try fallback to signature docstring
            if signature.__doc__ and signature.__doc__.strip():
                logger.warning(
                    "signature.instructions was empty, falling back to __doc__"
                )
                return signature.__doc__

            # Final fallback to DEFAULT_INSTRUCTIONS
            logger.warning(
                "Could not extract optimized instructions from module, "
                "using DEFAULT_INSTRUCTIONS"
            )
            return DEFAULT_INSTRUCTIONS

        except Exception as e:
            logger.error(
                f"Failed to extract optimized instructions: {e}. "
                "Using DEFAULT_INSTRUCTIONS."
            )
            return DEFAULT_INSTRUCTIONS

    def fit(
        self,
        labelled_passages: list[LabelledPassage] | None = None,
        validation_size: float = 0.3,
        min_passages: int = 10,
        max_passages: int | None = 100,
        enable_wandb: bool = False,
        mipro_num_candidates: int = 10,
        mipro_num_trials: int = 15,
        mipro_max_bootstrapped_demos: int = 0,
        mipro_max_labeled_demos: int = 5,
        **kwargs,
    ) -> "AutoLLMClassifier":
        """
        Optimize labelling guidelines using DSPy MIPRO with precision focus.

        Uses F0.5 score which weights precision 2x more than recall, suitable
        for applications where false positives are more costly than false negatives.

        Parameters
        ----------
        labelled_passages : list[LabelledPassage] | None, default=None
            Training data. If None, uses concept.labelled_passages
        validation_size : float, default=0.15
            Proportion to use for validation (default 0.15 for 85/15 split)
        min_passages : int, default=10
            Minimum required passages to perform optimization
        max_passages : int | None, default=100
            Maximum passages to use (randomly sampled if more available).
            Useful for faster experimentation. If None, uses all passages.
        enable_wandb : bool, default=False
            Whether to log optimization progress to W&B
        mipro_num_candidates : int, default=10
            Number of instruction candidates to generate (MIPRO param)
        mipro_num_trials : int, default=15
            Number of optimization trials to run (MIPRO param)
        mipro_max_bootstrapped_demos : int, default=0
            Max few-shot demos to bootstrap (set to 0 for instruction-only optimization)
        mipro_max_labeled_demos : int, default=5
            Max labeled demos to include in optimization
        **kwargs
            Additional keyword arguments (currently unused)

        Returns
        -------
        AutoLLMClassifier
            Self, with optimized instructions stored

        Raises
        ------
        ValueError
            If insufficient labelled passages provided
        """
        import dspy
        from dspy.teleprompt import MIPROv2

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
            create_passage_level_fbeta_metric,
        )

        # 1. Load and validate data
        passages = labelled_passages or self.concept.labelled_passages

        # Sample down if max_passages specified
        if max_passages is not None and len(passages) > max_passages:
            import random

            logger.info(
                f"Sampling {max_passages} passages from {len(passages)} available"
            )
            # Use random seed for reproducibility
            random.seed(self.random_seed)
            passages = random.sample(passages, max_passages)

        if len(passages) < min_passages:
            logger.warning(
                f"Only {len(passages)} labelled passages available for {self.concept.wikibase_id}. "
                f"At least {min_passages} required for optimization. "
                f"Falling back to default instructions."
            )
            self.optimized_instructions = DEFAULT_INSTRUCTIONS
            self.is_fitted = True
            return self

        # 2. Prepare train/val split
        train_examples, val_examples = self._prepare_dspy_examples(
            passages, validation_size
        )

        # 3. Configure DSPy with evaluation model (used for scoring instruction candidates)
        # Temperature > 0 ensures varied outputs across different instruction candidates
        # cache=False ensures each instruction candidate gets fresh LLM responses
        # Without this, DSPy might return cached responses that don't reflect
        # instruction differences, causing all candidates to score identically
        evaluation_lm = self._create_dspy_lm(
            model_name=self.evaluation_model_name, temperature=0.7
        )
        dspy.configure(lm=evaluation_lm, cache=False)

        # 4. Create proposal model for generating instruction candidates
        # This can be a smarter/more capable model since it's called fewer times
        proposal_lm = self._create_dspy_lm(
            model_name=self.proposal_model_name, temperature=1.0
        )

        logger.info(
            f"Optimization models - Evaluation: {self.evaluation_model_name}, "
            f"Proposal: {self.proposal_model_name}"
        )

        # 5. Create DSPy module for concept tagging task
        concept_tagger = ConceptTaggerModule(
            concept_description=self.concept.to_markdown(),
            signature=ConceptTaggingSignature,
        )

        # 6. Setup passage-level F-beta metric (F0.5) for precision-focused optimization
        # F0.5 weights precision 2x more than recall while maintaining F1 constraint
        assert self.concept.wikibase_id is not None
        metric_fn = create_passage_level_fbeta_metric(
            concept_id=self.concept.wikibase_id,
            beta=0.5,  # Precision-focused: weights precision 2x more than recall
        )

        # 7. Run MIPRO optimization
        wandb_run = None
        try:
            # Optional W&B integration
            if enable_wandb:
                import wandb

                wandb_run = wandb.init(
                    project=f"auto-llm-{self.concept.wikibase_id}",
                    config={
                        "classifier_model_name": self.model_name,
                        "evaluation_model_name": self.evaluation_model_name,
                        "proposal_model_name": self.proposal_model_name,
                        "concept_id": str(self.concept.wikibase_id),
                        "num_train": len(train_examples),
                        "num_val": len(val_examples),
                    },
                )

            optimizer = MIPROv2(
                metric=metric_fn,
                prompt_model=proposal_lm,  # Smart model for generating instruction candidates
                auto=None,  # Disable auto mode to manually control num_candidates
                num_candidates=mipro_num_candidates,
                max_bootstrapped_demos=0,  # Disable bootstrapping for instruction-only optimization
                max_labeled_demos=0,  # Disable labeled demos for instruction-only optimization
                init_temperature=1.0,
                verbose=True,
            )

            optimized_module = optimizer.compile(
                student=concept_tagger,
                trainset=train_examples,
                valset=val_examples,
                num_trials=mipro_num_trials,
                minibatch=False,  # Disable minibatch for small datasets
                requires_permission_to_run=False,
            )

            # 8. Extract optimized instructions
            self.optimized_instructions = self._extract_optimized_instructions(
                optimized_module
            )

            if enable_wandb and wandb_run is not None:
                wandb_run.log(
                    {
                        "optimized_instructions": self.optimized_instructions,
                        "final_system_prompt": self.system_prompt,
                    }
                )
                wandb_run.finish()

        except Exception as e:
            logger.error(f"Optimization failed: {e}. Using default instructions.")
            self.optimized_instructions = DEFAULT_INSTRUCTIONS
            if enable_wandb and wandb_run is not None:
                wandb_run.finish(exit_code=1)
            self.is_fitted = True
            return self

        # 9. Update system prompt with optimized instructions
        self.system_prompt_template = AUTO_DEFAULT_SYSTEM_PROMPT
        self.system_prompt = self.system_prompt_template.format(
            concept_description=self.concept.to_markdown(),
            instructions=self.optimized_instructions,
        )

        # 10. Recreate pydantic-ai agent with new prompt
        self.agent = self._create_agent()

        # 11. Mark as fitted
        self.is_fitted = True

        logger.info(
            f"Optimization complete for {self.concept.wikibase_id} "
            f"(using F0.5 metric for precision focus)."
        )
        logger.info(f"Optimized instructions:\n{self.optimized_instructions}")

        return self
