# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `poetry run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.


## Updating a Classifier

First we run the training scripts. This will upload the classifier to s3 and link to it from its weights and biases project. You can then run the promote script which is used for promoting a model to primary within an aws environment. Promotion as adds the model to the weights and bias registry and setting it as primary gives it the environment alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
just train "Q123" --track --upload --aws-env sandbox
```

Then we promote:

```shell
just promote "Q123" --classifier_id abcd2345 --aws-env sandbox --primary
```

You can also achieve the above directly with:

```shell
just deploy-classifiers sandbox Q374
```

Or run a batch sequentially with:

```shell
just deploy-classifiers sandbox "Q374 Q473"
```

This is useful when you are already resolved that the trained model will become the new primary.
