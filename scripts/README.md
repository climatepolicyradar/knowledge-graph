# Scripts

This directory contains scripts that are used to run various processes around the concept store / knowledge graph. Generally, they should be run from the root of the repository using `poetry run python scripts/<script_name>.py`, or using a specific `just` command (see [justfile](../justfile) for more details).

Individual scripts' docstrings contain more information about their purpose and usage.


## Updating a Classifier

First we run the training scripts. This will increment the version of the classifier in the projects registry in weights and biases. You can then run the promote script which is used for promoting a model to primary within or cross aws environment. Promotion will not affect the version / alias.

_Note: You will need a profile in your `.aws/config` file with an active terminal session to use the following command as the upload command requires s3 access._

```shell
(.venv) ➜  knowledge-graph git:(main) ✗ poetry run python scripts/train.py --wikibase-id "Q123" --track --upload --aws-env prod
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: cpr-mark (climatepolicyradar). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.7
wandb: Run data is saved locally in /Users/markcottam/PycharmProjects/knowledge-graph/wandb/run-20250416_095506-zpo111i7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run kind-night-7
wandb: ⭐️ View project at https://wandb.ai/climatepolicyradar/Q123
wandb: 🚀 View run at https://wandb.ai/climatepolicyradar/Q123/runs/zpo111i7
[10:04:04] Using next version v0                                                                                                                                                                             train.py:267
           Saved RulesBasedClassifier("Cliff stabilisation") to /Users/markcottam/PycharmProjects/knowledge-graph/data/processed/classifiers/Q123/qh2xjyvd.pickle                                            train.py:276
[10:04:05] Uploading RulesBasedClassifier to Q123/RulesBasedClassifier/v0/model.pickle in bucket cpr-prod-models                                                                                             train.py:159
[10:04:06] Uploaded RulesBasedClassifier to Q123/RulesBasedClassifier/v0/model.pickle in bucket cpr-prod-models                                                                                              train.py:169
wandb:
wandb: 🚀 View run kind-night-7 at: https://wandb.ai/climatepolicyradar/Q123/runs/zpo111i7
wandb: ⭐️ View project at: https://wandb.ai/climatepolicyradar/Q123
wandb: Synced 4 W&B file(s), 0 media file(s), 4 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250416_095506-zpo111i7/logs
```

Then we promote:

```shell
(.venv) ➜  knowledge-graph git:(main) ✗ poetry run python scripts/promote.py --wikibase-id "Q123" --classifier KeywordClassifier --version v3 --within-aws-env prod --primary
wandb: Using wandb-core as the SDK backend.  Please refer to <https://wandb.me/wandb-core> for more information.
wandb: Currently logged in as: cpr-mark (climatepolicyradar). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.7
wandb: Run data is saved locally in /Users/markcottam/PycharmProjects/knowledge-graph/wandb/run-20250416_101929-ci7w1o5z
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run winter-sponge-9
wandb: ⭐️ View project at <https://wandb.ai/climatepolicyradar/Q123>
wandb: 🚀 View run at <https://wandb.ai/climatepolicyradar/Q123/runs/ci7w1o5z>
wandb:
wandb: 🚀 View run winter-sponge-9 at: <https://wandb.ai/climatepolicyradar/Q123/runs/ci7w1o5z>
wandb: ⭐️ View project at: <https://wandb.ai/climatepolicyradar/Q123>
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250416_101929-ci7w1o5z/logs
```
