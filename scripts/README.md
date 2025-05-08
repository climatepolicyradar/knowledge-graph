# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `poetry run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.


## Updating a Classifier

First we run the training scripts. This will increment the version of the classifier in the projects registry in weights and biases. You can then run the promote script which is used for promoting a model to primary within or cross aws environment. Promotion will not affect the version / alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
just train "Q123" --track --upload --aws-env sandbox
```

Then we promote:

```shell
just promote "Q123" --classifier KeywordClassifier --version v3 --within-aws-env sandbox --primary
```

You can also achieve the above directly with:

```shell
just train-promote Q374 sandbox
```

Or run a batch sequentially with:

```shell
just train-promote-many "Q374 Q473" sandbox
```

This is useful when you are already resolved that the trained model will become the new primary.
