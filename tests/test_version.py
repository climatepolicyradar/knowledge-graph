import pytest

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.version import Version, get_latest_model_version


@pytest.mark.parametrize("input_version", ["v1", "v10", "v0"])
def test_version_valid(input_version):
    assert str(Version(input_version)) == input_version


@pytest.mark.parametrize("invalid_version", ["1", "v", "v1.0", "latest"])
def test_version_invalid(invalid_version):
    with pytest.raises(ValueError):
        Version(invalid_version)


@pytest.mark.parametrize(
    "version_a,version_b,expected",
    [
        ("v1", "v2", True),
        ("v2", "v1", False),
        ("v10", "v2", False),
    ],
)
def test_version_comparison(version_a, version_b, expected):
    assert (Version(version_a) < Version(version_b)) == expected


def test_version_sorting():
    versions = [Version("v3"), Version("v1"), Version("v10"), Version("v2")]
    sorted_versions = sorted(versions)
    assert [str(v) for v in sorted_versions] == ["v1", "v2", "v3", "v10"]


def test_version_sorting_with_larger_numbers():
    versions = [
        Version("v3"),
        Version("v1"),
        Version("v10"),
        Version("v2"),
        Version("v20"),
    ]
    sorted_versions = sorted(versions)
    assert [str(v) for v in sorted_versions] == ["v1", "v2", "v3", "v10", "v20"]


def test_version_latest_not_supported():
    with pytest.raises(ValueError, match="`latest` isn't yet supported"):
        Version("latest")


def test_version_equality():
    assert Version("v1") == Version("v1")
    assert Version("v1") == "v1"
    assert Version("v1") != Version("v2")
    assert Version("v1") != "v2"


def test_version_hash():
    versions = {Version("v1"), Version("v2"), Version("v1")}
    assert len(versions) == 2


def test_get_latest_model_version():
    class MockArtifact:
        def __init__(self, version, aws_env):
            self.version = version
            self.metadata = {"aws_env": aws_env.name}

    artifacts = [
        MockArtifact("v1", AwsEnv("labs")),
        MockArtifact("v2", AwsEnv("labs")),
        MockArtifact("v1", AwsEnv("staging")),
        MockArtifact("v2", AwsEnv("staging")),
        MockArtifact("v3", AwsEnv("labs")),
        MockArtifact("v6", AwsEnv("production")),
        MockArtifact("v8", AwsEnv("prod")),
    ]

    latest_version = get_latest_model_version(artifacts, AwsEnv.labs)
    assert str(latest_version) == "v3"
    assert type(latest_version) is Version

    latest_version_staging = get_latest_model_version(artifacts, AwsEnv.staging)
    assert str(latest_version_staging) == "v2"

    latest_version_prod = get_latest_model_version(artifacts, AwsEnv.production)
    assert str(latest_version_prod) == "v8"
