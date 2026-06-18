"""
Shared Snowflake connection logic: reusable, Prefect-free domain code.

Both `build_dataset` and `predict` operations query Snowflake, so the connection
logic lives here rather than in either one. When explicit key-pair credentials are
supplied (cloud/ECS path, resolved from SSM by the flows), uses the DbtBot service
account; otherwise falls back to the local `~/.snowflake/config.toml` connection.
"""

import snowflake.connector
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)

from knowledge_graph.utils import get_logger

SNOWFLAKE_WAREHOUSE = "PRODUCTION_DBT_WH"
SNOWFLAKE_DATABASE = "PRODUCTION"
SNOWFLAKE_SCHEMA = "PUBLISHED"


def connect_to_snowflake(
    snowflake_user: str | None = None,
    snowflake_private_key: str | None = None,
    snowflake_account: str | None = None,
):
    """
    Connect to Snowflake.

    When all three explicit credentials are supplied (cloud/ECS path), uses key-pair
    authentication with the DbtBot service account. When none are supplied, falls back
    to connection_name="local_dev" for local development.

    A partial set of credentials almost always indicates a misconfiguration (e.g. an
    empty SSM parameter), so it raises rather than silently falling back to local_dev
    and surfacing a misleading "config.toml not found" error.
    """
    # Normalise empty strings (e.g. an empty SSM parameter) to None so they're
    # treated as "not supplied" rather than as valid credentials.
    match tuple(
        value or None
        for value in (snowflake_user, snowflake_private_key, snowflake_account)
    ):
        case (str() as user, str() as private_key_pem, str() as account):
            private_key = load_pem_private_key(private_key_pem.encode(), password=None)
            private_key_bytes = private_key.private_bytes(
                encoding=Encoding.DER,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption(),
            )
            return snowflake.connector.connect(
                account=account,
                user=user,
                private_key=private_key_bytes,
                warehouse=SNOWFLAKE_WAREHOUSE,
                database=SNOWFLAKE_DATABASE,
                schema=SNOWFLAKE_SCHEMA,
            )

        case (None, None, None):
            # Local development fallback reads from ~/.snowflake/config.toml
            try:
                return snowflake.connector.connect(connection_name="local_dev")
            except snowflake.connector.errors.Error as e:
                get_logger().error(
                    "Failed to connect to Snowflake. "
                    f"Error: {e!r} "
                    "Ensure you have a config.toml generated. You can find one in your Snowflake account settings. "
                    "See https://docs.snowflake.com/en/developer-guide/snowflake-cli/connecting/configure-connections#define-connections"
                )
                raise

        case _:
            raise ValueError(
                "Partial Snowflake credentials supplied: snowflake_user, "
                "snowflake_private_key, and snowflake_account must all be provided "
                "together for key-pair authentication, or all omitted to use the local "
                "config.toml connection."
            )
