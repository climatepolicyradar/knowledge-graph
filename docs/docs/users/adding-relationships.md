# Adding relationships between concepts

Relationships between concepts are set as **statements** on the concept page.

Each statement is made up of an **item**, a **property**, and a **value**. For example, the statement `Extreme Cold` `Subconcept of` `Extreme Weather` (item, property, and value respectively) would be used to describe the relationship between two concepts. It would be displayed on the concept page for `Extreme Cold`.

NB Both concepts must exist in the concept store before you can add a relationship between them. If the concept you want to relate to doesn't exist yet, you'll need to [create it](./creating-a-new-concept.md) first.

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

## Editing relationships

To edit a relationship, click the `✏️ edit` button on the right hand side of the statement you want to change. Make the necessary changes in the form that appears, and click `✅ save` to save your changes. To abandon your changes, click `❌ cancel`.

## Deleting relationships

To delete a relationship,`✏️ edit` button on the right hand side of the statement you want to change, and then click the `🗑️ remove` button.

If you want to undo the deletion, go to the item's history, find the appropriate revision in the list, and click the corresponding `undo` button.
