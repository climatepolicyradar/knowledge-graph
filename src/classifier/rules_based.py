from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.concept import Concept
from src.span import Span


class RulesBasedClassifier(Classifier):
    """
    Classifier uses keyword matching to find instances of a concept in text.

    This classifier uses two KeywordClassifiers: one for the positive labels and one for
    the negative labels. It then filters out any positive matches which overlap with
    negative matches.

    For example, given a concept like:
        Concept(preferred_label="gas", negative_labels=["greenhouse gas"])
    the classifier will match
        "I need to fill up my gas tank"
    but not
        "The greenhouse gas emissions are a major contributor to climate change."
    """

    def __init__(self, concept: Concept):
        """
        Create a new RulesBasedClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)

        self.positive_matcher = KeywordClassifier(
            concept=Concept(
                wikibase_id=self.concept.wikibase_id,
                preferred_label=self.concept.preferred_label,
                alternative_labels=self.concept.all_labels,
            )
        )
        if self.concept.negative_labels:
            self.negative_matcher = KeywordClassifier(
                concept=Concept(
                    wikibase_id=self.concept.wikibase_id,
                    preferred_label=self.concept.negative_labels[0],
                    alternative_labels=self.concept.negative_labels,
                )
            )
        else:
            self.negative_matcher = None

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        positive_matches = self.positive_matcher.predict(text)
        negative_matches = (
            self.negative_matcher.predict(text) if self.negative_matcher else []
        )

        # filter out any positive matches which overlap with negative matches
        filtered_matches = [
            match
            for match in positive_matches
            if not any(
                match.overlaps(negative_match) for negative_match in negative_matches
            )
        ]

        # Update the labeller name on each span from KeywordClassifier to
        # RulesBasedClassifier
        for match in filtered_matches:
            match.labellers = [str(self)]

        return filtered_matches
