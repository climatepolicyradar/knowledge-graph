"""Wrapper around LLMClassifier which automatically improves its own prompt."""

import asyncio
from dataclasses import dataclass
from typing import Self

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from knowledge_graph.classifier.large_language_model import (
    LLMClassifier,
    LLMClassifierPrompt,
)
from knowledge_graph.identifiers import WikibaseID
from scripts.evaluate import (
    create_gold_standard_labelled_passages,
    create_validation_predictions_dataframe,
    evaluate_classifier,
)
from scripts.get_concept import get_concept_async

DEFAULT_META_PROMPT = """
You are an expert prompt engineer helping to improve a text classification system.

The classifier is tasked with identifying mentions of a specific concept in policy documents.
Below is the concept description:

<concept_description>
{concept_description}
</concept_description>

Labelling guidelines can be used to ensure this description is applied consistently. 
The current labelling guidelines (if any) are:
<current_guidelines>
{current_guidelines}
</current_guidelines>

Here are the validation results from the current classifier. Each row shows:
- The passage of text which was classified;
- Whether the gold standard says it contains the concept;
- Whether the classifier predicted it contains the concept;
- Whether the classifier's prediction was correct, based on the gold standard.

<validation_results>
{validation_results}
</validation_results>

Based on these results, generate improved labelling guidelines that will help 
the classifier perform better. 

First, take stock of what is working. Based on the CORRECT examples, which guidelines
are worth preserving? Keep this in mind as you suggest your changes.

Next, focus on the INCORRECT predictions. These should be the basis for the suggested changes.
In essence: where are the old guidelines lacking and which emergent rules from the gold labels 
should be made explicit to help the classifier perform? 
More practically, look for:
1. Patterns in false positives (classifier said yes, but should be no) - what should NOT be tagged
2. Patterns in false negatives (classifier said no, but should be yes) - what SHOULD be tagged
3. Edge cases and ambiguous situations


The guidelines should be clear, specific, and actionable. 
Use numbered lists to provide clarity and structure in the guidelines.
The guidelines should provide clear rules for inclusion and exclusion.
The guidelines will be used to make replicable decisions across many passages of text,
so they should generalise well and provide clarity in ambiguous situations.
NEVER use examples directly from the validation results to avoid overfitting to the validation set. 
Instead, add new examples you think illustrate the core decision well. Use these sparingly.
"""


class OptimiserResponse(BaseModel):
    """Response from the optimiser model containing new labelling guidelines."""

    analysis: str = Field(
        description="Brief analysis of the validation results, identifying patterns in errors"
    )
    labelling_guidelines: str = Field(
        description="Improved labelling guidelines as a numbered list"
    )


@dataclass
class TrialResult:
    """Result of a single optimisation trial."""

    prompt: LLMClassifierPrompt
    f_beta_score: float
    predictions_df: pd.DataFrame


class AutoLLMClassifier(LLMClassifier):
    """
    A classifier that automatically improves its own prompt through iterative optimisation.

    Uses a meta-prompt to analyse validation results and generate improved labelling guidelines.
    """

    def fit(
        self,
        labelled_passages: list | None = None,
        meta_prompt: str | None = None,
        optimiser_model_name: str = "openrouter:google/gemini-3-pro-preview",
        final_classifier_model_name: str | None = None,
        n_trials: int = 3,
        beta: float = 1.0,
        batch_size: int = 16,
        **kwargs,
    ) -> Self:
        """
        Iterate on a prompt for the underlying classifier.

        :param labelled_passages: no-op. Makes this interface compatible with the train
            script.
        :param meta_prompt: The meta-prompt for the optimiser model. If None, uses default.
        :param optimiser_model_name: The model name for the optimiser
        :param final_classifier_model_name: The model name for the final classifier.
            If None, uses the underlying classifier's model name.
        :param n_trials: Number of optimisation trials to run
        :param beta: Beta value for f-beta score calculation (default 1.0 = F1)
        :param batch_size: Batch size for evaluation
        :return: Self with updated prompt
        """
        if meta_prompt is None:
            meta_prompt = DEFAULT_META_PROMPT

        labelled_passages = self.concept.labelled_passages
        trial_results: list[TrialResult] = []

        print("=" * 60)
        print("Trial 0: Evaluating initial classifier")
        print("=" * 60)
        initial_result = self._evaluate_and_store(
            classifier=self,
            labelled_passages=labelled_passages,
            beta=beta,
            batch_size=batch_size,
        )
        trial_results.append(initial_result)
        print(f"Initial f-beta score (beta={beta}): {initial_result.f_beta_score:.4f}")

        optimiser_agent = Agent(
            model=optimiser_model_name,
            output_type=OptimiserResponse,
        )

        # Run optimisation trials
        current_prompt = self.system_prompt_template
        current_predictions_df = initial_result.predictions_df

        for trial in range(n_trials):
            print("=" * 60)
            print(f"Trial {trial + 1}/{n_trials}: Generating new guidelines")
            print("=" * 60)

            # Create new guidelines and prompt using optimiser model
            new_guidelines = self._generate_new_labelling_guidelines(
                optimiser_agent=optimiser_agent,
                meta_prompt=meta_prompt,
                current_prompt=current_prompt,
                predictions_df=current_predictions_df,
            )
            print(f"New guidelines:\n{new_guidelines}")

            new_prompt = LLMClassifierPrompt(
                system_prompt_template=current_prompt.system_prompt_template,
                labelling_guidelines=new_guidelines,
            )

            # Create and evaluate new classifier
            trial_classifier = LLMClassifier(
                concept=self.concept,
                model_name=self.model_name,
                system_prompt_template=new_prompt,
            )

            trial_result = self._evaluate_and_store(
                classifier=trial_classifier,
                labelled_passages=labelled_passages,
                beta=beta,
                batch_size=batch_size,
            )
            trial_results.append(trial_result)
            print(f"Trial {trial + 1} f-beta score: {trial_result.f_beta_score:.4f}")

            # Update for next iteration
            current_prompt = new_prompt
            current_predictions_df = trial_result.predictions_df

        best_trial = max(trial_results, key=lambda t: t.f_beta_score)
        best_index = trial_results.index(best_trial)
        print("=" * 60)
        print(
            f"Best trial: {best_index} with f-beta score: {best_trial.f_beta_score:.4f}"
        )
        print("=" * 60)

        # Create final classifier with the best prompt
        # Use self.model_name if final_classifier_model_name is not provided
        model_name = (
            final_classifier_model_name
            if final_classifier_model_name is not None
            else self.model_name
        )
        print("Creating final classifier with best prompt")
        final_classifier = LLMClassifier(
            concept=self.concept,
            model_name=model_name,
            system_prompt_template=best_trial.prompt,
        )

        # Evaluate final classifier
        print("=" * 60)
        print("Evaluating final classifier")
        print("=" * 60)
        final_result = self._evaluate_and_store(
            classifier=final_classifier,
            labelled_passages=labelled_passages,
            beta=beta,
            batch_size=batch_size,
        )
        print(
            f"Final classifier f-beta score (beta={beta}): {final_result.f_beta_score:.4f}"
        )

        # Update self with the best prompt
        self.system_prompt_template = best_trial.prompt
        self.system_prompt = best_trial.prompt.format(self.concept)
        self.agent = self._create_agent()

        return self

    def _evaluate_and_store(
        self,
        classifier: LLMClassifier,
        labelled_passages: list,
        beta: float,
        batch_size: int,
    ) -> TrialResult:
        """
        Evaluate a classifier and return a TrialResult.

        :param classifier: The classifier to evaluate
        :param labelled_passages: Labelled passages for evaluation
        :param beta: Beta value for f-beta score
        :param batch_size: Batch size for evaluation
        :return: TrialResult with prompt, score, and predictions
        """
        _, model_labelled_passages, confusion_matrix = evaluate_classifier(
            classifier=classifier,
            labelled_passages=labelled_passages,
            batch_size=batch_size,
        )

        f_beta_score = confusion_matrix.f_beta_score(beta=beta)
        gold_standard = create_gold_standard_labelled_passages(labelled_passages)
        predictions_df = create_validation_predictions_dataframe(
            gold_standard, model_labelled_passages
        )

        return TrialResult(
            prompt=classifier.system_prompt_template,
            f_beta_score=f_beta_score,
            predictions_df=predictions_df,
        )

    def _generate_new_labelling_guidelines(
        self,
        optimiser_agent: Agent[None, OptimiserResponse],
        meta_prompt: str,
        current_prompt: LLMClassifierPrompt,
        predictions_df: pd.DataFrame,
    ) -> str:
        """
        Generate new labelling guidelines using the optimiser model.

        :param optimiser_agent: The pydantic-ai agent for the optimiser
        :param meta_prompt: The meta-prompt template
        :param current_prompt: The current LLMClassifierPrompt
        :param predictions_df: DataFrame of validation predictions
        :return: New labelling guidelines string
        """

        incorrect_df = predictions_df[~predictions_df["correct"]]
        if len(incorrect_df) == 0:
            # If all correct, show a sample of correct predictions
            validation_results = predictions_df.head(10).to_string()
        else:
            validation_results = incorrect_df.to_string()

        formatted_meta_prompt = meta_prompt.format(
            concept_description=self.concept.to_markdown(),
            current_guidelines=current_prompt.labelling_guidelines or "None",
            validation_results=validation_results,
        )

        optimiser_response = optimiser_agent.run_sync(formatted_meta_prompt)

        print(f"Optimiser analysis:\n{optimiser_response.output.analysis}")

        return optimiser_response.output.labelling_guidelines


if __name__ == "__main__":
    concept = asyncio.run(
        get_concept_async(
            wikibase_id=WikibaseID("Q715"),
            include_labels_from_subconcepts=True,
            include_recursive_has_subconcept=True,
        )
    )

    clf = AutoLLMClassifier(
        concept,
        model_name="openrouter:google/gemini-3-flash-preview",
    )

    clf.fit(
        optimiser_model_name="openrouter:google/gemini-3-flash-preview",
        final_classifier_model_name="openrouter:google/gemini-3-pro-preview",
        n_trials=5,
        beta=0.5,
    )
