# Classifier types

> **NOTE:** Users of the concept store deserve a thorough explanation of how their contributions are used in downstream services (like classifiers), but the process for structuring and distinguishing between each type of classifier is still in active development. This page is therefore a work-in-progress/placeholder. When we've developed a clearer strategy for how we define and build classifiers, I'll update this page with clearer guidance.

We use data from the concept store to build a variety of types of concept classifier. The major classes of classifier that we're developing are:

## Keyword classifiers

These classifiers are a simple keyword search through our policy text for mentions of the preferred label and alternative labels for each concept. Any text which mentions any of these keywords can be classified as an instance of the concept.

**Example:** A concept for `Greenhouse gases` might include the alternative labels `gases`, `ghg`, `ghgs`, `gas`. Using a keyword classifier, any text which mentions any of the concept's preferred- or alternative labels could be classified as an instance of the `Greenhouse gases` concept.

NB. The 'preferred label' and 'alternative labels' on a concept aren't treated differently by a keyword classifier. The classifier will pick out any of the concept's labels in the text, regardless of whether they're preferred or alternative. The only distinction is that the preferred label will be used wherever the concept in displayed in the app

## Rule-based classifiers

Using additional rules defined in the concept's statements, we can build classifiers which are a bit more sophisticated than simple keyword searches. For example, a concept might be keyword based, but should not include certain additional keywords.

**Example:** A concept for `Oil` (subconcept of `Fossil fuels`) might include the alternative labels `oil`, `petroleum`, `crude oil`, but might also specify a rule which states that the text should _not_ begin with the keywords `essential`, `olive`, `cooking`, `palm`, etc. This would allow us to exclude mentions of olive oil, palm oil, etc.

## Example-based classifiers

These are the more sophisticated, neural-network based models, which rely on large amounts of labelled data. A strict labelling strategy and concept definition should be developed for these classifiers, and a large number of positive and negative examples should be collected through our labelling tool. These classifiers are more powerful, and should be more robust to variations in language and context.

**Example:** A more nebulous concept like `Climate change monitoring` might be difficult to define with a simple keyword search, or rules. Instead, we might use a large number of positive examples (text which is definitely about climate change monitoring, despite not using those precise words) and negative examples (text which is definitely _not_ about climate change monitoring, even though it uses phrases like `climate change` or `monitoring`). Using some fancy machine learning magic, we can train a model to classify new text as an instance of `Climate change monitoring` or not.

## Hierarchy in classifiers

We know that concepts can have hierarchical relationships with one another. For example, `Extreme Cold` might be listed as a subconcept of `Extreme Weather`. When we build classifiers, we should be able to use this structure to our advantage.

If we have developed a classifier for `Extreme Cold`, and we know (through our concept hierarchy) that `Extreme Cold` is part of `Extreme Weather`, we can label any passage which matches our `Extreme Cold` classifier as an instance of _both_ `Extreme Cold` and `Extreme Weather`.
