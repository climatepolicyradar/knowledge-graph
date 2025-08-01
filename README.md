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

## The basics

### Concepts

Concepts are the core building blocks of our knowledge graph. They represent key ideas, terms, or topics which are important to understanding the climate policy domain. Each concept has a preferred label, optional alternative labels (synonyms, acronyms, related terms), a description, and can be linked to other concepts through hierarchical or associative relationships.

```python
from src.concept import Concept

extreme_weather_concept = Concept(
    preferred_label="extreme weather",
    description="it's like weather, but too much!!",
    alternative_labels=["extreme weather events", "rapid-onset events", "weather anomalies"],
)
```

Most of CPR's concepts are defined in our [concept store](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q374) (and are thus associated with a `wikibase_id`), but you can always define your own concepts in code.

### Classifiers and spans

Classifiers are used to identify concepts in text. We use a variety of classifier architectures throughout our knowledge graph, from basic keyword matching to more sophisticated BERT-sized models, to optimised calls to third-party LLMs.

Each classifier is single-class, meaning there's a 1:1 mapping between a `Concept` and a `Classifier`. When you call the `predict` method on a classifier with some input text, it returns a list of `Span` objects which indicate where the concept is mentioned.

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

Our `LabelledPassage` objects combine a passage of text with the spans that mention a particular concept. They can contain multiple spans, referring to multiple concepts, each labelled through a different method.

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

## So what is the knowledge graph?

We've built our knowledge graph by running a set of classifiers over our giant corpus of climate-relevant text.

In the short-term, identifying where each concept is mentioned in our documents makes it easier for interested users of CPR's tools to jump straight to the relevant sections of our documents.

In the longer term, we expect the graph to be a useful artefact in its own right. By analysing the structured web of relationships between climate policy concepts and the documents that mention them, we should be able to identify emerging topics and high-leverage areas for policy intervention.

## Testing

### Local Prequisites

* Install [Git LFS](https://git-lfs.com)
* Start Docker (Desktop) locally and follow instructions in [Vespa README.md](./tests/local_vespa/README.md)

To run the tests and exclude known problematic tests in the CI environment run

``just test -v -m "'not flaky_on_ci and not transformers'" ``

## Pipelines

Within this Knowledge Graph repo we have a full pipeline at `flows/full_pipeline.py:full_pipeline` that brings together three distinct steps into one parent pipeline. This is to enable a fully automated end to end run.

This solution calls inference -> aggregation -> indexing in series for all documents in the run as opposed to running single documents through concurrently. Eg. we wait for all inference jobs to complete before progressing. This was chosen for simplicity and to rely on the concurrency functionality and limits already integrated in to the sub flows / pipelines.

For example, we don't want to try and index 25k docs all at once and already have functionality for managing this within the indexing flow. To do this at the document level in this pipeline would require a lot more work relative to just calling the indexing flow from the parent flow.

All the sub pipelines (inference, aggregation & indexing) can be run individually as distinct steps.

1. Inference

This consists of running our classifiers over our documents to generate `LabelledPassages` which themselves contain `Spans` as listed above in this `README.md`. 

2. Aggregation

This consists of aggregating (collating) the inference results for a document from different classifiers which are stored at multiple s3 paths into one object in s3.

3. Indexing

This consists of indexing the spans identified from inference in to our passage index's concepts field and concept counts to our family index within our vespa database.
