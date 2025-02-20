# 🕸️ Knowledge graph

This repo comprises the infrastructure, tools, and scripts for managing Climate Policy Radar's concept store and knowledge graph.

## Getting started

This repo is orchestrated with a [justfile](./justfile) (see [just](https://github.com/casey/just)). To install the dependencies, run:

```bash
just install
```

You can see the full list of `just` commands by running:

```bash
just --list
```

## Basics

### Concepts

The knowledge graph is built from a collection of concepts. Concepts can have a preferred label, description, and alternative labels.

```python
from src.concept import Concept

extreme_weather_concept = Concept(
    preferred_label="extreme weather",
    description="it's like weather, but too much!!",
    alternative_labels=["extreme weather events", "rapid-onset events", "weather anomalies"],
)
```

Most of CPR's concepts are defined in our [concept store](https://climatepolicyradar.wikibase.cloud), but you can always define your own in code.

### Classifiers

Classifiers are used to identify concepts in text. We use a variety of classifier architectures throughout our knowledge graph, from basic keyword matching to more sophisticated BERT-sized models, to optimised calls to third-party LLMs.

Classifiers are single-class, ie there is a 1:1 mapping between a `Concept` and a `Classifier`. Calling the `predict` method on a classifier with some input text will return a list of `Spans` in which the concept is mentioned.

```python
from src.classifier import KeywordClassifier

extreme_weather_classifier = KeywordClassifier(concept=extreme_weather_concept)

predicted_spans = extreme_weather_classifier.predict("This is a passage of text about extreme weather")
```

```python
[
    Span(
        text='This is a passage of text about extreme weather',
        start_index=32,
        end_index=47,
        concept_id=None,
        labellers=['KeywordClassifier("extreme weather")'],
        id='sg5u338r',
        labelled_text='extreme weather'
    )
]
```

### Labelled passages

Labelled passages store a passage of text, together with the spans of text that match a particular concept. They can contain multiple spans for multiple concepts.

```python
from src.labelled_passage import LabelledPassage

labelled_passage = LabelledPassage(
    text="This is a passage of text about extreme weather",
    spans=[
        Span(
            text='This is a passage of text about extreme weather',
            start_index=32,
            end_index=47,
            concept_id=None,
            labellers=['KeywordClassifier("extreme weather")'],
            id='sg5u338r',
            labelled_text='extreme weather'
        )
    ],
)
```

Labelled passages are a versatile data structure that we use in many ways. They store the predictions from our classifiers, but passages labelled by human experts can also be used to train new models, or be used as a source of truth for comparing and evaluating the performance of candidate models.

## How does that all of that add up to a knowledge graph?

We've built our knowledge graph by running a set of classifiers over our giant corpus of climate-relevant text.

In the short-term, identifying where each concept is mentioned in our documents makes it easier for interested users to jump straight to the relevant sections of our documents.

In the longer term, we expect the graph to be a useful artefact in its own right. By analysing the structured web of relationships between climate policy concepts and the documents that mention them, we should be able to identify emerging topics and high-leverage areas for policy intervention.
