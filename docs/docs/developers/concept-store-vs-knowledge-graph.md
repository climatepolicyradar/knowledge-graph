# Concept Store vs Knowledge Graph

## What is the concept store?

Climate Policy Radar's **concept store** is a [Wikibase](https://www.mediawiki.org/wiki/Wikibase) instance. It's used by the policy team to manage individual climate concepts and the relationships between them.

The concepts are richly described within a common data model, which allows us to store information like the concept's name, definition, aliases, and its relationships to other concepts. Each concept is also assigned a unique identifier (eg `Q123`), which allows us to reference individual concepts in other parts of the system.

The policy team also use [Argilla](https://argilla.io/) to annotate text from real documents with individual concepts.

Based on that data, we can train classifiers to automatically recognise instances of our concepts in our documents. We use the labelled passage datasets from Argilla to evaluate the performance of our classifiers and to improve the quality of the data in the concept store.

The concept store is effectively a [CMS](https://en.wikipedia.org/wiki/Content_management_system) for the knowledge graph, ie. it's a user-friendly interface for managing (some of) the data in the knowledge graph.

## What is the knowledge graph?

The **knowledge graph** is a superset of the concept store, which includes additional data and relationships that are not directly related to the concepts themselves.

For example, we might store a concept for `flooding` and another for `extreme weather`, and then link them together with a relationship statement like `flooding` `is a subconcept of` `extreme weather`.

We can also track the relationships between concepts and documents. For example, we might annotate a particular passage in a document, saying that it mentions a particular concept. For example, the sentence "The flooding was caused by extreme weather" would be annotated with both the `flooding` and `extreme weather` concepts.

At scale, the network of relationships adds up to much more than the sum of its parts. We can use the graph to power new search and discovery features for users, or to generate new insights about the relationships between different climate concepts.

The graph doesn't exist in a single place, but is instead a collection of data stored throughout the pipeline and exposed selectively to users. The [concept store](#what-is-the-concept-store) is the primary interface for managing and interacting with the knowledge graph.
