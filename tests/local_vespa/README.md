# Local Vespa 

## Setup for tests

This folder has the config for a local vespa instance that is used for testing.

Some tests require this in order to run. You can spin up a local version by running:

```shell
just vespa_dev_setup
```

This requires the [vespa cli](https://docs.vespa.ai/en/vespa-cli.html). It also expects the docker daemon to be running.

## Skip vespa tests instead

If you'd rather avoid this you can also run pytest without vespa using the following:

```shell
just test-without-vespa
```
