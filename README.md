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
from knowledge_graph.concept import Concept

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
from knowledge_graph.classifier import KeywordClassifier

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
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span

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

### Local Prerequisites

* Install [Git LFS](https://git-lfs.com)
* Start Docker (Desktop) locally and follow instructions in [Vespa README.md](./tests/local_vespa/README.md)

To run the tests

`just test -v`

If you experience test failures for target transformer tests in [test_targets.py](./tests/test_targets.py) you can exclude them by running

`just test -v -m "'not transformers'"`

## Pipelines

Within this Knowledge Graph repo we have a full pipeline at [flows/full_pipeline.py:full_pipeline](./flows/full_pipeline.py) that brings together three distinct steps into one parent pipeline. This is to enable a fully automated end to end run.

This solution calls inference -> aggregation -> indexing in series for all documents in the run as opposed to running single documents through concurrently. Eg. we wait for all inference jobs to complete before progressing. This was chosen for simplicity and to rely on the concurrency functionality and limits already integrated in to the sub flows / pipelines.

For example, we don't want to try and index 25k docs all at once and already have functionality for managing this within the indexing flow. To do this at the document level in this pipeline would require a lot more work relative to just calling the indexing flow from the parent flow.

All the sub pipelines (inference, aggregation & indexing) can be run individually as distinct steps.

1. Inference

This consists of running our classifiers over our documents to generate `LabelledPassages` which themselves contain `Spans` as listed above in this `README.md`.
The list of classifiers  and rules under which they are run is defined in [classifier_spec yaml file for each environment](./flows/classifier_specs/v2) e.g. the [production.yaml](./flows/classifier_specs/v2/production.yaml)
2. Aggregation

This consists of aggregating (collating) the inference results for a document from different classifiers which are stored at multiple s3 paths into one object in s3.

3. Indexing

This consists of indexing the spans identified from inference in to our passage index's concepts field and concept counts to our family index within our vespa database.

## Deployment and flows with Prefect

Prefect Deployments are defined in [deployments.py](./deployments.py)

A push of a commit to a PR will deploy to Sandbox environment

A merge to `main` branch will deploy the to the Labs, Staging, and Production environments

You may also run the Github Actions Workflow directly in the [Github UI to deploy to an environment such as Sandbox](https://github.com/climatepolicyradar/knowledge-graph/actions/workflows/prefect_deploy_sandbox.yml)

For [full details see the Notion page](https://www.notion.so/climatepolicyradar/KG-Deployment-Triggering-2799109609a48043bcd5fdc77df1c94d)

### Monitoring Deployment status

The [Prefect Dashboard shows all deployments, flows and runs](https://app.prefect.cloud/account/4b1558a0-3c61-4849-8b18-3e97e0516d78/workspace/1753b4f0-6221-4f6a-9233-b146518b4545/deployments?g_range={%22type%22:%22span%22,%22seconds%22:-2592000}),
which you can filter by name and date of last deployment.

The name of the knowledge graph deployments all have the same prefix of `kg-` and they are programmatically generated following the format:

`kg-<flow_name>-<environment>`

**or**

`kg-full-pipeline-<env>`

Flows can be run via in [/flows](./flows)

## Scripts

All helper scripts are located in [/scripts directory](./scripts) directory

## Classifier training, promotion and deployment

These are performed via helper scripts run by the [justfile](./justfile) commands. These are currently executed manually on a local laptop are are not part of a CI/CD pipeline.

They can be found in the [/scripts directory](./scripts) directory

Here is the implementation of Classifier [training](https://github.com/climatepolicyradar/knowledge-graph/blob/main/scripts/train.py), [promotion](https://github.com/climatepolicyradar/knowledge-graph/blob/main/scripts/promote.py) and [deployment](https://github.com/climatepolicyradar/knowledge-graph/blob/main/scripts/deploy.py) processes.

## Static sites

We have several [static sites](./static_sites/) which can be generated from the outputs of the Knowledge Graph

* [labelling_librarian](./static_sites/labelling_librarian/)
* [concept_librarian](./static_sites/concept_librarian/)

These can be created by the `justfile` commands

### Run a static site locally

`just serve-static-site vibe_check`

### Generate a static site

`just generate-static-site vibe_check`

## Concept Store

The concept store is an internal tool used to structure and manage key concepts in climate policy. It helps power automated concept detection in our datasets. More can be read about this at Climate Policy Radar's wikibase cloud [domain](https://climatepolicyradar.wikibase.cloud/wiki/Main_Page).

#### User Management

Users can be managed within Climate Policy Radar's organisation under the `SpecialPages` tab. We have taken an approach of rolling users passwords after they stop contributing to keep audit history of their edits within the Concept Store.

#### Programmatic Access

To access Wikibase Cloud we have created _bot passwords_ with basic privileges such that programmatic access is not linked to real users.
