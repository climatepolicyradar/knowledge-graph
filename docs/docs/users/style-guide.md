# Style guide

This style guide should inform how concepts are written/formatted in the concept store.

In most cases, the following guidelines are not strict rules. They are intended to help you make decisions about how to structure your concepts, but they are not exhaustive, and they are not always the best way to structure your concepts. If you have a good reason to break these rules, you should!

If you do break these guidelines, it's worth recording your reasoning in the corresponding edit message or on the item's [talk page](talk-pages.md).

## Preferred labels

[Wikidata's guidance on labels](https://www.wikidata.org/wiki/Help:Label)

Labels for items in the concept store should:

### Refer to one thing and one thing only

Our concepts should be atomic. We should therefore avoid creating concepts that include words like 'and' or 'or'.

For example, a concept like `barriers and challenges` should probably be broken into two separate concepts, `barriers` and `challenges`.

Some important concepts known by a non-atomic name in the climate policy community will break this rule, e.g., `loss and damage` ([see below](#reflect-common-usage))

### Reflect common usage

Preferred labels should make it easy for readers to understand the meaning of the concept at a glance, and should reflect its most common usage in the climate policy community.

- the preferred label for a species should use its common name, while its scientific name should be added to the list of [alternative labels](#alternative-labels).
- the concept `loss and damage` is a well-known concept in the climate policy community, and so should be used as the preferred label for a single concept instead of being broken into two separate concepts.

### Should not contain caveats

The names of our concepts should be direct and free of caveats or equivocations. If you feel like your concept _does_ need these caveats, it's worth considering whether you can break it down into a group of smaller concepts, or whether the structure of its neighbouring concepts needs to change.

- `Technologies (adaptation)` should be `adaptation technologies`
- `Clean technologies (general)` should be `clean technologies`

### Should be lowercase

Labels should begin with a lowercase letter, except in cases where uppercase is normally required or expected.

Pretend the label appears in the middle of a normal sentence and follow standard language rules. In regular text, only proper nouns such as the names of specific people, places, specific organisations, etc., would be capitalized.

- `carbon tax`
- `greenhouse gas`
- `Paris Agreement`

### Should not contain symbols

We should prefer whole words over symbols, eg `loss and damage` over `loss & damage`

### Should not contain abbreviations

We should prefer whole words over abbreviations, eg `carbon dioxide` over `CO2`

### Should be singular

We should lean towards singular, eg

- `storm surge` over `storm surges`
- `tornado` over `tornadoes`

### Should be gender neutral

We should avoid gendered language, unless gender is an intrinsic part of that concept. Gendered forms of the word should be added to the concept as [alternative labels](#alternative-labels).

- `firefighter` over `fireman`

## Alternative labels

[Wikidata's guidance on alternative labels](https://www.wikidata.org/wiki/Help:Aliases)

Alternative labels should include:

- **synonyms** eg a concept for `child` should contain the alternative label `youth`
- **plural forms** eg a concept for `greenhouse gas` should contain the alternative label `greenhouse gases`
- **extra spaces** eg a concept for `greenhouse gas` should contain the alternative label `green house gas`
- **scientific abbreviations** eg a concept for `flu` should contain the alternative label `influenza`
- **combinations of the above!** eg a concept for `child` might contain the alternative labels `children`, `youth`, `kid`, and `kids` (ie a plural form of another alternative label)

Alternative labels should not include:

- **differences in casing** eg a concept for `greenhouse gas` should not contain both alternative labels `GHG` and `ghg`
- **differences in punctuation** eg a concept for `capacity building` should not contain the alternative label `capacity-building`
- **typos or misspellings** eg a concept for `carbon dioxide` should not contain the alternative label `carbon dioxyde`

### Should meet users' expectations when searching

Users should be able to search for an alternative label and see that concept in their search results, eg a search for `kids` might return the `child` concept.

If your alternative label isn't a direct synonym for your concept, users of our search tools might find the difference between their query and the returned concept's preferred label jarring.
In these situations, consider starting a new topic on the item's [talk page](talk-pages.md) about creating a new subconcept or sibling concept to encompass the alternative label.

As a result of the guidance above, we should find that alternative labels will added much more liberally for leaf concepts than for root/branch concepts.

### Can overlap with the alternative labels of another concept

We can return multiple concept results in search, so it's fine for alternative labels to overlap with the alternative labels for another concept.

## Descriptions

[Wikidata's guidance on descriptions](https://www.wikidata.org/wiki/Help:Description)

### Should be sufficient to disambiguate the concept

Descriptions should provide enough information to disambiguate a concept with an ambiguous preferred label. Wikidata gives a nice example here:

> [London (Q92561)](https://www.wikidata.org/wiki/Q92561) has the description "city in Southwestern Ontario, Canada"

This is enough to disambiguate the concept from the first `London` that comes to mind, which is probably the capital of the UK. London, Ontario could be described in much more detail, but the description gives us just enough information to disambiguate it.

The brevity of our descriptions will naturally make them a bit coarse and reductive, but that's okay! We should aim to provide an exhaustive definition of the concept in its [Definition](#definitions)

### Should avoid language/information that is likely to change

Words and phrases like "current", "expected", or "last year's" will eventually need to be changed. We won't always be able to catch these changes in time, so we should endeavour to write descriptions which will _remain_ true.

## Definitions

Definitions go beyond descriptions. While a [description](#descriptions) should disambiguate a concept at a glance, a definition should enable readers to identify instances of the concept in any text.

### Should be exhaustive

Definitions should be part of the public record of what we believe the concept is, and we should be prepared for external users of our tools to question our definitions. A complete definition should cover all edge cases and exceptions to the concept.

### Be sufficient to label examples of the concept in text

Definitions should be precise enough that a reader could identify instances of the concept in a passage of text during a labelling task. For example, a definition of `greenhouse gas` should be sufficient to identify instances of `carbon dioxide`, `methane`, and `nitrous oxide` in a given text.

Note that the definition of a concept will almost always be iteratively developed, based on many edge cases discovered during labelling. It's okay for the first pass at a definition to be a bit coarse or gappy, as long as we have time to update it!

### Should be supported by evidence

Definitions should be based on the best available evidence, and should be supported by [references](#references) where possible.

### Also avoid information that is likely to change

As above, words and phrases like "current", "expected", or "last year's" in the definition will eventually need to be changed. We won't always be able to catch these changes in time, so we should endeavour to write definitions which will _remain_ true.

## References

References should be used to justify your editing decisions to your fellow editors. They might pre-empt a controversial edit, or be added as part of a resolution of a discussion on the item's [talk page](talk-pages.md).

### Should appear at statement-level, not item-level

References should be added to individual statements on an item, instead of the item as a whole.

The difference is similar to the difference between a bibliography and in-line citations used in academic writing:

- Item-level references are like a bibliography: You have to read the whole thing to know what the references are there to prove, and even then it's not always clear! If an update is made to a statement on the item by another editor, it's not clear which references are still relevant.
- Statement-level references are like in-line citations: you can see exactly what each reference is there to prove, and can quickly verify the information you're reading.

### Are nice to have, but not essential

References are not essential for every statement in the concept store. They are most useful when you're adding a statement which is likely to be challenged, or is founded on a substantial amount of research.

### Can be supported by a `Quotation`

A supporting `Reference` will be easier to understand if it's accompanied by a direct `Quotation` which supports that statement. A `Quotation` gives readers a clearer indication of the useful information contained in the source, guarding against a [telephone game](https://en.wikipedia.org/wiki/Chinese_whispers) of misinterpretation.

Storing a relevant `Quotation` also means that even in cases where a referenced URL breaks, your peers will still be able to understand the information you're trying to convey.

### Won't be used in downstream services

References are for internal editor use only, and won't be used in downstream services. For example, while we might use the relationship between concepts to inform the training of our machine learning models, we won't use the references which support those relationships. The same goes for the any `Quotation` which accompanies a reference.

### Can be added in groups

Feel free to add multiple references to a single statement if you're drawing on multiple sources to inform your edits.

### Should be URLs

References should be URLs to the source of the information you're using to inform your edits. This is the most useful form of reference for your peers, as it allows them to quickly verify the information you're using to inform your edits. We currently don't have a way to store references which aren't URLs.

### Can link to non-academic sources

References don't need to be academic sources. They can be to any source which you think will help your peers understand your edits. This might include:

- academic papers
- news articles
- reputable blog posts
- internal documentation
- slack conversations
- notion pages
- etc.
