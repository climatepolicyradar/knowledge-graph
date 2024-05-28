# Concept heuristics

Where should we draw the line between concepts? How do we know when a hierarchy has become too granular? How do we know when we've reached the useful edge of our knowledge graph?

## Where's the edge of our domain / knowledge graph?

We're not aiming to build a perfect model of the world with our knowledge graph. We should represent the world of climate policy as accurately as possible, but we should also be pragmatic about the limitations of our tools, our knowledge, and the utility of those things to our users!

We don't want to spend all our time building a perfect model of the world, only to find that it's not useful to anyone. We should aim to build a model that's _just_ good enough to be useful, and then iterate on it as we learn more about what our users need.

## What makes a good concept?

- Concepts should generally represent a single idea.
- Root concepts (ie those with no `Subconcept of` relationships) should be very broad, and represent a large field of knowledge, eg `Just transition`, or `Technologies`. These should form the very base of the hierarchies within the knowledge graph, and should have many connections to other concepts.
- Leaf concepts (ie those with no `Subconcept` relationships) should be very specific, and represent a single, atomic idea, eg `Carbon dioxide`, or `United Kingdom`. These should be the outermost edges of the hierarchies within the knowledge graph.
- Intermediate concepts (sometimes referred to as "branch concepts") should be somewhere in between, and should represent a single, coherent idea that can be broken down into smaller parts. As they form the middle of our hierarchy of abstraction, they should be the easiest to connect across hierarchies. They should have the most relationships to other concepts, with a mix of various `Subconcept of`, `Has subconcept`, and `Related` relationships.
- Concepts should be easy to relate to other concepts. If you're struggling to find a way to link a concept to the rest of the knowledge graph, it might be too granular or removed from the climate policy domain.
- Concepts should be useful to our users (and beyond). If a concept is only understandable by the experts in our team, we should try to abstract/generalise it until it's useful to a wider audience.

## How do we know when a hierarchy is too granular?

See [Hierarchy heuristics](hierarchy-heuristics.md) for more thoughts on this question.
