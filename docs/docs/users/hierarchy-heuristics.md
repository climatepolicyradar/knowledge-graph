# Hierarchy heuristics

This page provides some rough guidance on how to think about hierarchies in the concept store. Like everything else in this documentation, these are just suggestions. If you have a good reason to go against this guidance (and have discussed it with the team), you should! If that new insight applies to more than a couple of specific cases, you should add it to this documentation so that it's easier for the next person to get started.

## Hierarchies should be used to represent levels of abstraction

When considering new subconcept relationships, we should ensure that all subconcepts can be treated as instances of the parent concept. That is, a parent concept should be a single level of abstraction above its subconcepts.

Structuring hierarchies in this way will allow us to build classifiers which can treat mentions of their subconcepts as mentions of themselves. This should greatly increase the number of concept mentions in the app, and consequently improve our users' ability to find relevant information.

For example, if we have developed a classifier for `Extreme Cold`, and we know (through our concept hierarchy) that `Extreme Cold` is part of `Extreme Weather`, we can label any passage which matches our `Extreme Cold` classifier as an instance of _both_ `Extreme Cold` and `Extreme Weather`.

Users should see mentions of `Extreme Cold` when they're searching mentions of for `Extreme Weather`, but not vice versa.

## We should avoid being too hierarchical

As we're developing our concept store and bringing structure to our knowledge graph, we should try to avoid relying too heavily on hierarchical relationships. While hierarchy is useful (especially as a starting point for new fields), we should lean more on "relatedness" than "hierarchy" for a few reasons:

1. **Hierarchies are hard to maintain:** The strict parent-child relationships in a hierarchy can be hard to maintain. The real world is messy, and most concepts don't fit neatly into one single category. It's almost impossible to represent levels of abstraction consistently across fields/disciplines in a hierarchical way (many librarians have tried!), and it's even harder to maintain these levels of abstraction as the world changes. This is particularly true in fields like climate and policy, where the meaning of a concept can drift in a matter of years.  
   Taking a more flexible, networked approach to concepts' relationships allows us to represent the complexity of the real world more accurately, and makes it easier to adapt to new information or changes in the world.

2. **Hierarchies are difficult to navigate:** Users who are clicking around [climatepolicyradar.org](https://climatepolicyradar.org/) might use the links between concept pages (ie, their relationships) to find what they're looking for. It's really difficult for a non-expert user to anticipate how a concept might be categorised into our specific hierarchies, and getting lost in trees of irrelevant categories with no path to where you want to go can be frustrating!  
   Non-hierarchical relationships allows us to make connections _between_ different hierarchies, fields, and disciplines, giving users an escape hatch from an irrelevant field. We should provide users as many paths between concepts as we can, and make it as easy as possible for them to find what they're looking for. Rather than defining strict family trees, we want a web of relationships with only a few degrees of separation between any two concepts. We should aim to build a network like wikipedia's, where every page is only a few clicks away from every other.

3. **Hierarchies are prone to bias:** Even when it's not deliberate, hierarchies make it very easy for their authors to reinforce existing power structures in the real world that they're trying to represent. They often end up marginalising or excluding already-marginalised groups or ideas, because they're difficult to incorporate into the limited number of categories or strata available.  
   By allowing for a broader range of relationships between concepts, we'll make it easier for ourselves to incorporate a broader range of perspectives and ideas, and to avoid accidentally reinforcing existing biases.

4. **Hierarchies aren't very interesting:** The limitations of hierarchy aren't just boring for users, they're boring for machines too! A graph of relationships between concepts is a much more complex and interesting foundation for data science and machine learning work than a strict tree. The benefit of building a richer network of concepts should become even more pronounced as the number of concepts in the system grows, and the _value_ of those cross-cutting relationships will also compound as we leverage the knowledge graph to provide more sophisticated search and recommendation features.

## We should use hierarchies as a starting point

While we should avoid being _too_ hierarchical, we can still use hierarchies as a starting point for exploring new fields. Deliberately oversimplifying the world into a hierarchy can be a useful way to get the ball rolling, and our workflow means it's always possible to refine the structure later.

When we're starting a new field, we should try to find a high-level concept that captures the essence of the field, and then add subconcepts that are part of that high-level concept. Examples might include 'Mitigation', 'Technologies', 'Policy Instruments', 'Sectors', etc.

We've also seen that other organisations often arrange their concepts in hierarchical taxonomies. Bootstrapping our work with their knowledge should speed up our classifier development, and make it easier to collaborate with those orgs.

Nevertheless, we should be prepared to adjust the hierarchical representations where they break down, and be quick to add `Related` relationships from those newly-inherited concepts to others in our existing network.

## How deep should our hierarchies be?

We should aim to keep our hierarchies relatively shallow. Doing so will (hopefully) keep the concepts abstract enough to be useful to users, while making it easier for us to maintain the hierarchy as the world changes.

The concepts inhereted from [GST](https://gst1.org) worked over three levels of abstraction, and we should aim to keep our future hierarchies at a similar depth.

## Hierarchies can overlap

Although a hierarchy will usually start life as a single tree, it's likely that as we develop our concept store, we'll find that concepts might exist in multiple hierarchies, with overlapping branches. This is fine! We should aim to represent the relationships between concepts as accurately as possible, and if that means that a concept is a subconcept of two different parent concepts, that's okay.

Take a look at [the documentation on merging concepts](./merging-existing-concepts.md) for more information on how merge two concepts from different hierarchies which represent the same thing.

## FAQ

### Can a concept have multiple parents?

Yes! A concept can technically have multiple parent concepts. This is a common pattern in the real world, where a concept might be a subconcept of multiple parent concepts. However, you should consider whether the concept is really a subconcept of both parent concepts, or whether it's actually `Related` to one of them.

For example, we might consider `Drought` to be a subconcept of both `Extreme Weather` and `Water Scarcity`. However, this might better be represented as `Drought` being a subconcept of `Extreme Weather`, and `Water Scarcity` being `Related` to `Drought`.

There are some instances where a concept is a subconcept of multiple parent concepts, but they should be used carefully.

### Can a concept's grandparent also be its parent?

Technically, yes. But just because it _can_ happen, doesn't mean it _should_.

In almost all cases, the hierarchical relationships between concepts should be as simple as possible, with one parent concept and multiple subconcepts.
Before you decide to create a relationship structure like this, you should consider:

- Whether the child concept is the subconcept of both parent concepts (as above), but the parent concepts are not subconcepts of each other
- Whether the child concept is actually `Related` to one of the parent concepts (see above).
- Whether the intermediate parent concept is actually unnecessary

There are some instances where these relationships make sense. For example, [Greenhouse gases](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q218) have subconcepts [Ozone-Depleting Substances](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q235), [Hydrochlorofluorocarbons](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q240), and [Chlorofluorocarbons](https://climatepolicyradar.wikibase.cloud/wiki/Item:Q237) among others, but Hydrochlorofluorocarbons and Chlorofluorocarbons are also subconcepts of Ozone-Depleting Substances. This sort of relationship is acceptable, but should be used sparingly.
