# Local Vespa

## Setup for tests

This folder has the config for a local Vespa instance that is used for testing.

Some tests require this in order to run. You can spin up a local version by running:

```shell
just vespa_dev_setup
```

This requires the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html). It also expects the Docker daemon to be running.

## Skip vespa tests instead

If you'd rather avoid this you can also run pytest without vespa using the following:

```shell
just test-without-vespa
```


## Additional Query Profiles

The `additional_query_profiles` subdirectory contains query profiles that are copied into the local vespa instance's application package for the duration of the tests as we need additional ones to facilitate unit testing.

This is done at test time as we automatically update the application package under the `test_app` directory should there be any changes to the application package in the `infra` repo in github. 
