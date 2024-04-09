# Merging concepts

When two concepts in wikibase represent the same thing, they can be merged. This will usually improve the quality of the network of concepts in the graph, and should be a common practice with a big enough graph.

## When should concepts _not_ be merged?

It's surprisingly tricky to decide when two concepts represent the same thing! The guidance on [the wikidata Help:Merge page](https://www.wikidata.org/wiki/Help:Merge) offers a few useful examples:

>en:tree → de:Baum (German word for tree)  
>Merge? ✅  
>Reason: These are about the same thing (tree) and so should be merged into the >(multilingual) item, tree (Q10884)
>
>en:tree → de:Eiche (German word for oak tree)  
>Merge? ❌  
>Reason: These are different things (oak tree is a "subclass of" tree) and so should >remain separate as two items; tree (Q10884) and oak (Q33036816), respectively
>
>Miller (family name) → Miller (disambiguation page)  
>Merge? ❌  
>Reason: These are different concepts, even if some Wikipedia articles include family names on disambiguation pages. They should remain separate as two items; Miller >(Q1605060) and Miller (Q304896), respectively

If two concepts are about _exactly_ the same thing, they should be merged. If not, using a "Related to" relationship might be more appropriate! You might also be able to represent a hierarchy between the concepts (as in the `Eiche` `subclass of` `Tree` example above). If you're not sure, start a discussion on the item's talk page!

## How to merge concepts

- Identify the two concepts that should be merged
- Remove any statements which link the items together (eg `Related to`, `Subconcept of`, `Has subconcept`) from both items. If the items are linked, the merge will fail.
- Go to [the concepts store's Special:MergeItems page](https://climatepolicyradar.wikibase.cloud/wiki/Special:MergeItems)
- Enter the two concept IDs and click "Merge"
