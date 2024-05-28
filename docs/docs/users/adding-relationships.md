# Adding relationships between concepts

Relationships between concepts are set as **statements** on the concept page.

Each statement is made up of an **item**, a **property**, and a **value**. For example, the statement `Extreme Cold` `Subconcept of` `Extreme Weather` (item, property, and value respectively) would be used to describe the relationship between two concepts. It would be displayed on the concept page for `Extreme Cold`.

TODO: each statement needs to have a type specified before its value can be entered

## Relationship types

There are three major types of relationship which you can use to describe the how concepts are connected to one another. These are:

### Subconcept of

This relationship is used to describe a hierarchical relationship between two concepts. For example, `Extreme Cold` is a subconcept of `Extreme Weather`.

### Has subconcept

This relationship is used to describe a hierarchical relationship between two concepts. For example, `Extreme Weather` has a subconcept `Extreme Cold`. It's the inverse of the `Subconcept of` relationship.

### Related to

This relationship is used to describe a non-hierarchical relationship between two concepts. For example, `Bushfire reduction` is related to `Forest management`.

### Thinking about hierarchy

Hierarchies are complicated! There's some guidance on how to think about hierarchies in the [hierarchy heuristics](./hierarchy-heuristics.md) documentation. Make sure you understand the implications of adding a new relationship before you dive in!

## Adding a new relationship

To add a new statement, click the `+ add statement` button in the top right hand side of the item content.

![](./images/edit-item.png)

A form will appear where you can add a new statement. The form will prompt you for the following information:

- **Property:** The property that describes the relationship between the two concepts. You can search for the property you want to use by starting to type. The value should be one of [the properties in the concept store](https://climatepolicyradar.wikibase.cloud/wiki/Special:ListProperties) (aka the listed relationship types above).
- **Value:** The concept that the item is related to. You can search for the concept you want to use by starting to type.

You might be prompted to add a [qualifier](https://www.wikidata.org/wiki/Help:Qualifiers) or [rank](https://www.wikidata.org/wiki/Help:Ranking) to the statement. These are optional, and we aren't currently using them in the concept store.

## References

TODO: Move this to <references.md>

Each statement can be supported by an optional reference. To add a reference, click the `+ add reference` button on the right hand side of the statement you want to reference. The value should be a URL to a reliable source that supports the statement. That might be something direct and formal (eg a scientific paper, a news article, a government report), or indirect (eg a conversation on slack, or a page in notion). As long as the reference offers sufficient explanation for why a decision was made, it's a good reference.

## Editing relationships

To edit a relationship, click the `✏️ edit` button on the right hand side of the statement you want to change. Make the necessary changes in the form that appears, and click `✅ save` to save your changes. To abandon your changes, click `❌ cancel`.

## Deleting relationships

To delete a relationship,`✏️ edit` button on the right hand side of the statement you want to change, and then click the `🗑️ remove` button.

If you want to undo the deletion, go to the item's history, find the appropriate revision in the list, and click the corresponding `undo` button.

## TODO: Concepts need to be created before they can be added in statements
