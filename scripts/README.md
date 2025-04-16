# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `poetry run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.


## Updating a Classifier

First we run the training scripts. This will increment the version of the classifier in the projects registry in weights and biases. You can then run the promote script which is used for promoting a model to primary within or cross aws environment. Promotion will not affect the version / alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
poetry run python scripts/train.py --wikibase-id "Q123" --track --upload --aws-env prod
```

Then we promote:

```shell
just promote "Q123" --classifier KeywordClassifier --version v3 --within-aws-env prod --primary
```
