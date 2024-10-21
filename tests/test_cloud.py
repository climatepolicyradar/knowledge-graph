from scripts.cloud import AwsEnv


def test_init_awsenv():
    assert AwsEnv.staging == AwsEnv("dev")
