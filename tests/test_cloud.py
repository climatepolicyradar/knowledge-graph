from unittest.mock import Mock, patch

import botocore
import pytest
import yaml
from moto import mock_aws
from prefect.variables import Variable

from knowledge_graph.cloud import (
    AwsEnv,
    ClassifierSpec,
    Compute,
    function_to_flow_name,
    generate_default_job_variables_name,
    generate_deployment_name,
    get_prefect_job_variable,
    get_prefect_job_variables,
    is_logged_in,
    parse_aws_env,
    parse_spec_file,
)


def test_function_to_flow_name():
    assert function_to_flow_name(is_logged_in) == "is-logged-in"


def test_init_awsenv():
    assert AwsEnv.staging == AwsEnv("dev")


def test_prod_awsenv():
    assert AwsEnv.production == AwsEnv("prod")
    aws_envs = [AwsEnv("prod"), AwsEnv("production")]
    for aws_env in aws_envs:
        assert aws_env.name == "production"
        assert aws_env.value == "prod"


@pytest.mark.parametrize(
    "aws_env, use_aws_profiles, is_logged_in_result",
    [
        (AwsEnv.labs, True, True),
        (AwsEnv.sandbox, True, False),
        (AwsEnv.staging, True, True),
        (AwsEnv.production, True, False),
        (AwsEnv.labs, False, True),
        (AwsEnv.sandbox, False, False),
    ],
)
@mock_aws
def test_is_logged_in(aws_env, use_aws_profiles, is_logged_in_result):
    with patch("knowledge_graph.cloud.get_sts_client") as mock_get_sts_client:
        mock_sts = Mock()
        mock_get_sts_client.return_value = mock_sts

        if is_logged_in_result:
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        else:
            mock_sts.get_caller_identity.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "GetCallerIdentity",
            )

        assert is_logged_in(aws_env, use_aws_profiles) == is_logged_in_result

        expected_aws_env = aws_env if use_aws_profiles else None
        mock_get_sts_client.assert_called_once_with(expected_aws_env)
        mock_sts.get_caller_identity.assert_called_once()


@pytest.mark.parametrize(
    "invalid_input",
    ["invalid", "test"],
)
def test_parse_aws_env_invalid(invalid_input):
    with pytest.raises(ValueError, match=f"'{invalid_input}' is not one of"):
        parse_aws_env(invalid_input)


@pytest.mark.parametrize(
    "flow_name, aws_env, expected",
    [
        ("inference", AwsEnv.labs, "kg-inference-labs"),
        ("aggregate", AwsEnv.sandbox, "kg-aggregate-sandbox"),
        ("index", AwsEnv.staging, "kg-index-staging"),
        ("full-pipeline", AwsEnv.production, "kg-full-pipeline-prod"),
    ],
)
def test_generate_deployment_name(flow_name, aws_env, expected):
    assert generate_deployment_name(flow_name, aws_env) == expected


@pytest.mark.parametrize(
    "spec_contents,expected_specs",
    [
        # Test valid single entry
        (["Q123:v1"], [ClassifierSpec(name="Q123", alias="v1")]),
        # Test valid multiple entries
        (
            ["Q123:v1", "Q456:v2"],
            [
                ClassifierSpec(name="Q123", alias="v1"),
                ClassifierSpec(name="Q456", alias="v2"),
            ],
        ),
        # Test empty list
        ([], []),
    ],
)
def test_parse_spec_file(spec_contents, expected_specs, tmp_path):
    # Create a temporary spec file
    test_env = AwsEnv.sandbox
    spec_dir = tmp_path / "classifier_specs"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / f"{test_env}.yaml"

    with open(spec_file, "w") as f:
        yaml.dump(spec_contents, f)

    # Patch the SPEC_DIR to use our temporary directory
    with patch("knowledge_graph.cloud.SPEC_DIR", spec_dir):
        result = parse_spec_file(test_env)
        assert result == expected_specs


@pytest.mark.parametrize(
    "invalid_contents",
    [
        ["invalid_format"],  # Missing colon separator
        ["Q123:v1:extra"],  # Too many separators
        ["Q123"],  # No version
        [":v1"],  # No name
        ["Q123:"],  # No version after separator
    ],
)
def test_parse_spec_file_invalid_format(invalid_contents, tmp_path):
    # Create a temporary spec file
    test_env = AwsEnv.sandbox
    spec_dir = tmp_path / "classifier_specs"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / f"{test_env}.yaml"

    # Write test contents
    import yaml

    with open(spec_file, "w") as f:
        yaml.dump(invalid_contents, f)

    # Patch the SPEC_DIR to use our temporary directory
    with patch("knowledge_graph.cloud.SPEC_DIR", spec_dir):
        with pytest.raises(ValueError):
            parse_spec_file(test_env)


def test_compute_str():
    assert str(Compute.CPU) == "cpu"


@pytest.mark.parametrize(
    "compute, aws_env, expected",
    [
        (Compute.CPU, AwsEnv.labs, "cpu-default-job-variables-prefect-mvp-labs"),
        (Compute.GPU, AwsEnv.staging, "gpu-default-job-variables-prefect-mvp-staging"),
        (Compute.CPU, AwsEnv.production, "cpu-default-job-variables-prefect-mvp-prod"),
    ],
)
def test_generate_default_job_variables_name(compute, aws_env, expected):
    assert generate_default_job_variables_name(compute, aws_env) == expected


@pytest.mark.asyncio
async def test_get_prefect_job_variables():
    test_var_name = "cpu-default-job-variables-prefect-mvp-labs"
    await Variable.set(
        test_var_name, {"KEY1": "value1", "KEY2": "value2"}, overwrite=True
    )

    result = await get_prefect_job_variables(Compute.CPU, AwsEnv.labs)
    assert result == {"KEY1": "value1", "KEY2": "value2"}


@pytest.mark.asyncio
async def test_get_prefect_job_variables_not_found():
    with pytest.raises(ValueError, match="Variable '.*' not found in Prefect"):
        await get_prefect_job_variables(Compute.CPU, AwsEnv.staging)


@pytest.mark.asyncio
async def test_get_prefect_job_variable():
    test_var_name = "cpu-default-job-variables-prefect-mvp-labs"
    await Variable.set(
        test_var_name, {"KEY1": "value1", "KEY2": "value2"}, overwrite=True
    )

    result = await get_prefect_job_variable("KEY1", AwsEnv.labs, Compute.CPU)
    assert result == "value1"
