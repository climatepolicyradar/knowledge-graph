"""
DSPy components for AutoLLMClassifier optimization.

This module contains DSPy-specific components used by the AutoLLMClassifier
to optimize labelling guidelines for concept detection in policy documents.
"""

import logging
from typing import Callable

import dspy
from pydantic import ValidationError

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import count_passage_level_metrics
from knowledge_graph.span import Span, SpanXMLConceptFormattingError

logger = logging.getLogger(__name__)


class ConceptTaggingSignature(dspy.Signature):
    """
    DSPy signature for concept tagging task.

    Given a passage of text and a concept definition, identify whether
    the passage contains any mentions of the concept.

    You must reproduce the passage text exactly, adding <concept> XML tags
    around any mentions. If no mentions exist, return the text unchanged.

    CRITICAL: The output must be valid XML with balanced <concept> tags.
    """

    passage_text: str = dspy.InputField(
        desc="The passage text to analyze for concept mentions"
    )

    marked_up_text: str = dspy.OutputField(
        desc="The input text with <concept> tags added around mentions, "
        "or the exact original text if no mentions found"
    )


class ConceptTaggerModule(dspy.Module):
    """
    DSPy module for concept tagging task.

    This module wraps the concept tagging signature and prepends the concept
    description to the input, mirroring the structure of the pydantic-ai system prompt.

    Parameters
    ----------
    concept_description : str
        Markdown representation of the concept (from concept.to_markdown())
    signature : type[dspy.Signature]
        The DSPy signature class to use (ConceptTaggingSignature)
    """

    def __init__(self, concept_description: str, signature: type[dspy.Signature]):
        super().__init__()
        self.concept_description = concept_description
        self.predict = dspy.Predict(signature)

    def forward(self, passage_text: str) -> dspy.Prediction:
        """
        Run prediction on a passage.

        Parameters
        ----------
        passage_text : str
            Input passage text

        Returns
        -------
        dspy.Prediction
            Prediction with marked_up_text field
        """
        # Prepend concept description to passage
        # This mirrors what we do in pydantic-ai system prompt
        full_input = f"""Concept:
{self.concept_description}

Passage:
{passage_text}"""

        prediction = self.predict(passage_text=full_input)
        return prediction


def create_passage_level_f1_metric(concept_id: WikibaseID) -> Callable:
    """
    Create a passage-level F1 metric function for DSPy optimization.

    This wraps the existing count_passage_level_metrics() function from
    knowledge_graph.metrics to compute F1 score over batches of predictions.

    Parameters
    ----------
    concept_id : WikibaseID
        Concept ID to filter spans by

    Returns
    -------
    Callable
        Metric function with signature (examples, predictions, trace) -> float

    Notes
    -----
    DSPy optimizers expect a metric function that takes a batch of examples
    and predictions, not single examples. This aggregates F1 across the batch.
    """

    def passage_level_f1_metric(
        examples: list[dspy.Example] | dspy.Example,
        predictions: list[dspy.Prediction | str] | dspy.Prediction | str,
        trace=None,
    ) -> float:
        """
        Compute passage-level F1 score using existing metrics implementation.

        Parameters
        ----------
        examples : list[dspy.Example] | dspy.Example
            Ground truth examples with gold_spans and passage_id fields
        predictions : list[dspy.Prediction | str] | dspy.Prediction | str
            Model predictions with marked_up_text field or raw strings
        trace : Any, optional
            Optional trace (unused, required by DSPy signature)

        Returns
        -------
        float
            F1 score (0.0 to 1.0)
        """
        try:
            # Handle single example/prediction (convert to list)
            if not isinstance(examples, list):
                examples = [examples]
            if not isinstance(predictions, list):
                predictions = [predictions]

            # Convert examples to LabelledPassage (ground truth)
            ground_truth_passages = []
            for example in examples:
                # Validate example has required attributes
                if not hasattr(example, "passage_id"):
                    logger.debug(
                        f"Skipping example without passage_id. Type: {type(example)}"
                    )
                    continue

                ground_truth_passages.append(
                    LabelledPassage(
                        id=example.passage_id,
                        text=example.passage_text,
                        spans=example.gold_spans
                        if hasattr(example, "gold_spans")
                        else [],
                        metadata={},
                    )
                )

            # If no valid examples, return 0.0
            if not ground_truth_passages:
                logger.warning("No valid examples to evaluate, returning F1=0.0")
                return 0.0

            # Convert predictions to LabelledPassage
            predicted_passages = []
            for example, prediction in zip(examples, predictions):
                # Skip if example is invalid
                if not hasattr(example, "passage_id"):
                    continue

                # Extract marked_up_text from prediction (handle both Prediction objects and strings)
                if isinstance(prediction, str):
                    marked_up_text = prediction
                elif hasattr(prediction, "marked_up_text"):
                    marked_up_text = prediction.marked_up_text
                else:
                    logger.debug(f"Prediction has unexpected type: {type(prediction)}")
                    marked_up_text = ""

                # Parse predicted spans from XML
                try:
                    predicted_spans = Span.from_xml(
                        xml=marked_up_text,
                        concept_id=concept_id,
                        labellers=["dspy_optimizer"],
                        input_text=example.passage_text,
                    )
                except (SpanXMLConceptFormattingError, ValidationError) as e:
                    logger.debug(f"Failed to parse prediction XML: {e}")
                    predicted_spans = []
                except Exception as e:
                    logger.debug(
                        f"Failed to parse prediction with unexpected exception type: {e}"
                    )
                    predicted_spans = []

                predicted_passages.append(
                    LabelledPassage(
                        id=example.passage_id,
                        text=example.passage_text,
                        spans=predicted_spans,
                        metadata={},
                    )
                )

            # Use existing count_passage_level_metrics function
            cm = count_passage_level_metrics(
                ground_truth_passages=ground_truth_passages,
                predicted_passages=predicted_passages,
            )

            # Return F1 score from confusion matrix
            return cm.f1_score()

        except Exception as e:
            # Catch-all to prevent optimization from failing due to metric errors
            logger.error(f"Metric evaluation failed: {e}. Returning F1=0.0")
            return 0.0

    return passage_level_f1_metric


def create_passage_level_fbeta_metric(
    concept_id: WikibaseID,
    beta: float = 1.0,
) -> Callable:
    """
    Create DSPy metric function that computes passage-level F-beta score.

    This metric function is used by DSPy optimizers (like MIPRO) to evaluate
    predictions during prompt optimization. It compares predicted concept tags
    against ground truth labels at the passage level using F-beta score.

    The F-beta score allows tuning the precision/recall tradeoff:
    - beta < 1: Favor precision (e.g., 0.5 weights precision 2x more)
    - beta = 1: Standard F1 (equal weight)
    - beta > 1: Favor recall (e.g., 2 weights recall 2x more)

    Parameters
    ----------
    concept_id : WikibaseID
        The concept ID to filter spans for when computing metrics
    beta : float, default=1.0
        Weight of precision vs recall in F-beta calculation

    Returns
    -------
    Callable
        A metric function compatible with DSPy optimizers that takes
        (example, prediction, trace) and returns float in [0, 1]

    Notes
    -----
    The metric:
    1. Converts DSPy examples to LabelledPassage objects with ground truth spans
    2. Parses predicted XML markup into Span objects
    3. Uses count_passage_level_metrics() for binary classification evaluation
    4. Returns F-beta score from the confusion matrix

    Examples
    --------
    >>> # Precision-focused optimization (standard for precision tasks)
    >>> metric_fn = create_passage_level_fbeta_metric(
    ...     concept_id=WikibaseID("Q123"),
    ...     beta=0.5,
    ... )
    >>> # Use with MIPRO optimizer
    >>> optimizer = MIPROv2(metric=metric_fn, ...)
    """

    def metric(
        examples: list[dspy.Example] | dspy.Example,
        predictions: list[dspy.Prediction | str] | dspy.Prediction | str,
        trace=None,
    ) -> float:
        """
        Compute passage-level F-beta score for DSPy prediction(s).

        Parameters
        ----------
        examples : list[dspy.Example] | dspy.Example
            Ground truth example(s) with passage_text, passage_id, gold_spans
        predictions : list[dspy.Prediction | str] | dspy.Prediction | str
            Predicted markup (with <concept> tags) or list of predictions
        trace : Any, optional
            DSPy trace object (unused but required by DSPy interface)

        Returns
        -------
        float
            F-beta score in range [0, 1]
        """
        try:
            # Handle both single and batch inputs
            if not isinstance(examples, list):
                examples = [examples]
            if not isinstance(predictions, list):
                predictions = [predictions]

            # === STEP 1: Convert DSPy examples to LabelledPassage ground truth ===
            ground_truth_passages = []
            for example in examples:
                # Skip examples without passage IDs (shouldn't happen but be defensive)
                if not hasattr(example, "passage_id") or example.passage_id is None:
                    logger.warning(
                        f"Example missing passage_id field, skipping: {example}"
                    )
                    continue

                ground_truth_passages.append(
                    LabelledPassage(
                        id=example.passage_id,
                        text=example.passage_text,
                        spans=example.gold_spans
                        if hasattr(example, "gold_spans")
                        else [],
                        metadata={},
                    )
                )

            # If no valid examples, return 0.0
            if not ground_truth_passages:
                logger.warning("No valid examples to evaluate, returning F-beta=0.0")
                return 0.0

            # === STEP 2: Parse predictions into LabelledPassage objects ===
            predicted_passages = []
            for example, prediction in zip(examples, predictions):
                # Skip if example is invalid
                if not hasattr(example, "passage_id"):
                    continue

                # Extract marked_up_text from prediction
                if isinstance(prediction, dspy.Prediction):
                    marked_up_text = prediction.marked_up_text
                elif isinstance(prediction, str):
                    marked_up_text = prediction
                else:
                    logger.warning(
                        f"Unexpected prediction type: {type(prediction)}, skipping"
                    )
                    continue

                # Parse XML tags into Span objects
                try:
                    predicted_spans = Span.from_xml(
                        xml=marked_up_text,
                        concept_id=concept_id,
                        labellers=["dspy_optimizer"],
                        input_text=example.passage_text,
                    )
                except (SpanXMLConceptFormattingError, ValidationError) as e:
                    logger.debug(
                        f"Failed to parse prediction XML for passage {example.passage_id}: {e}"
                    )
                    predicted_spans = []
                except Exception as e:
                    logger.debug(
                        f"Unexpected error parsing prediction for passage {example.passage_id}: {e}"
                    )
                    predicted_spans = []

                predicted_passages.append(
                    LabelledPassage(
                        id=example.passage_id,
                        text=example.passage_text,
                        spans=predicted_spans,
                        metadata={},
                    )
                )

            # === STEP 3: Compute passage-level confusion matrix ===
            cm = count_passage_level_metrics(
                ground_truth_passages=ground_truth_passages,
                predicted_passages=predicted_passages,
            )

            # Return F-beta score instead of F1
            return cm.f_beta_score(beta=beta)

        except Exception as e:
            logger.error(f"Failed to compute F-beta metric: {e}. Returning 0.0")
            return 0.0

    return metric
