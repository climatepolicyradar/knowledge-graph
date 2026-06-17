"""
Shared Snowflake credential resolution for flows.

Flows that query Snowflake (build_dataset, predict_documents) resolve the DbtBot
service-account key-pair credentials from SSM here, then pass them into the
Prefect-free operations in `knowledge_graph.operations`.
"""

import base64

from pydantic import SecretStr

from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param

SNOWFLAKE_ACCOUNT_SSM = "/Snowflake/Account"
SNOWFLAKE_USER_SSM = "/Snowflake/ServiceUser/DbtBot/User"
SNOWFLAKE_PRIVATE_KEY_SSM = "/Snowflake/ServiceUser/DbtBot/PrivateKey"


def get_snowflake_credentials(aws_env: AwsEnv) -> tuple[str, str, SecretStr]:
    """
    Resolve Snowflake key-pair credentials from SSM.

    Returns (snowflake_account, snowflake_user, snowflake_private_key).
    """
    snowflake_account = get_aws_ssm_param(SNOWFLAKE_ACCOUNT_SSM, aws_env=aws_env)
    snowflake_user = get_aws_ssm_param(SNOWFLAKE_USER_SSM, aws_env=aws_env)
    snowflake_private_key = SecretStr(
        base64.b64decode(
            get_aws_ssm_param(SNOWFLAKE_PRIVATE_KEY_SSM, aws_env=aws_env)
        ).decode("utf-8")
    )

    return snowflake_account, snowflake_user, snowflake_private_key
