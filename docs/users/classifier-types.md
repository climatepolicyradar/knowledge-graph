# Classifier types

> **NOTE:** Users of the concepts store deserve a thorough explanation of how their contributions are used in downstream services (like classifiers), but the process for structuring and distinguishing between each type of classifier is still in active development. This page is therefore a work-in-progress/placeholder. When we've developed a clearer strategy for how we define and build classifiers, I'll update this page with clearer guidance.

We use data from the concepts store to build a variety of types of concept classifier. The major classes of classifier that we're developing are:

## Keyword classifiers

These classifiers are a simple keyword search through our policy text for mentions of the preferred label and alternative labels for each concept. Any text which mentions any of these keywords can be classified as an instance of the concept.

## Rule-based classifiers

Using additional rules defined in the concept's statements, we can build classifiers which are a bit more sophisticated than simple keyword searches. For example, a concept might be keyword based, but should not include certain additional keywords.

## Example-based classifiers

These are the more sophisticated, neural-network based models, which rely on large amounts of labelled data. A strict labelling strategy and concept definition should be developed for these classifiers, and a large number of positive and negative examples should be collected through our labelling tool. These classifiers are more powerful, and should be more robust to variations in language and context.

## Hierarchy in classifiers

We know that concepts can have hierarchical relationships with one another. For example, `Extreme Cold` might be listed as a subconcept of `Extreme Weather`. 

If we have developed a classifier for `Extreme Cold`, and we know (through our concept hierarchy) that `Extreme Cold` is part of `Extreme Weather`, we can label any passage which matches our `Extreme Cold` classifier as an instance of _both_ `Extreme Cold` and `Extreme Weather`.
