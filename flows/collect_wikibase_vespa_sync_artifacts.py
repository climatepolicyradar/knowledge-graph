import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from prefect import flow, task
from prefect.artifacts import acreate_table_artifact
from prefect.cache_policies import NONE
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import (
    ArtifactFilter,
    ArtifactFilterFlowRunId,
    ArtifactFilterKey,
    FlowFilter,
    FlowFilterId,
    FlowFilterName,
    FlowRunFilter,
    FlowRunFilterStartTime,
    FlowRunFilterState,
    FlowRunFilterStateType,
)
from prefect.client.schemas.objects import Artifact, FlowRun
from prefect.client.schemas.sorting import FlowRunSort
from prefect_gcp import GcpCredentials
from pydantic import BaseModel, Field

from flows.utils import get_logger
from flows.wikibase_to_vespa import wikibase_to_vespa
from knowledge_graph.cloud import AwsEnv


class ConceptSyncRecord(BaseModel):
    """
    A single sync event record from a flow run artifact.

    Represents one row from the raw artifact data. Multiple ConceptSyncRecords
    can exist for the same concept if it was synced across different flow runs.

    Example: If concept Q123 was synced 3 times, there will be 3 ConceptSyncRecords.
    """

    concept_id: str = Field(description="The content-based concept ID")
    wikibase_id: str = Field(description="The Wikibase ID (e.g., Q123)")
    preferred_label: str = Field(description="The preferred label for the concept")
    wikibase_revision: str | int = Field(description="The Wikibase revision number")
    status: str = Field(description="Status: ✓ for success, ✗ for failure")
    error: str = Field(description="Error message if failed, N/A otherwise")
    flow_run_id: str = Field(description="The Prefect flow run ID")
    flow_run_name: str = Field(description="The Prefect flow run name")
    synced_at: datetime = Field(description="Timestamp when the artifact was created")


class MergedConceptRecord(BaseModel):
    """
    An aggregated view of a unique concept across all sync events.

    Represents one unique (Wikibase ID + Concept ID) combination with aggregated
    metadata from all syncs. This is what gets written to Google Sheets.

    Example: If concept Q123 was synced 3 times, there will be 1 MergedConceptRecord
    with sync_count=3 and the latest status/revision.
    """

    concept_id: str = Field(description="The content-based concept ID")
    wikibase_id: str = Field(description="The Wikibase ID (e.g., Q123)")
    preferred_label: str = Field(description="The preferred label for the concept")
    latest_wikibase_revision: str | int = Field(
        description="The most recent Wikibase revision"
    )
    latest_status: str = Field(description="Status from most recent sync")
    latest_error: str = Field(description="Error from most recent sync if any")
    sync_count: int = Field(description="Number of times this concept was synced")
    latest_sync_at: datetime = Field(description="Timestamp of most recent sync")
    latest_flow_run_name: str = Field(description="Most recent flow run name")
    all_flow_runs: list[str] = Field(
        description="All flow run names that synced this concept"
    )


@task(cache_policy=NONE)
async def get_wikibase_to_vespa_flow_runs(
    aws_env: AwsEnv,
    lookback_days: int = 30,  # This is Prefect's amount for our plan
) -> list[FlowRun]:
    """
    Get all Wikibase → Vespa flow runs

    Only for a specific AWS environment.
    """
    logger = get_logger()

    async with get_client() as client:
        # Calculate the lookback time
        lookback_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # First, find the sync flow itself
        flows = await client.read_flows(
            flow_filter=FlowFilter(name=FlowFilterName(any_=[wikibase_to_vespa.name]))
        )

        if not flows:
            logger.warning("no flow found")
            raise ValueError("TODO")

        if len(flows) > 1:
            logger.error(
                f"found too many matching flows: {','.join(list(map(lambda f: f.name or 'no name', flows)))}"
            )
            raise ValueError("TODO")

        flow = flows[0]
        logger.info(f"found flow: {flow.name} (id: {flow.id})")

        # Query for flow runs from this flow with pagination
        all_flow_runs = []
        offset = 0
        page_size = 200  # Prefect has a max limit of 200 per request

        while True:
            flow_runs_page = await client.read_flow_runs(
                flow_filter=FlowFilter(id=FlowFilterId(any_=[str(flow.id)])),
                flow_run_filter=FlowRunFilter(
                    start_time=FlowRunFilterStartTime(after_=lookback_time),
                    # Only get completed runs (successful or failed, but not running)
                    state=FlowRunFilterState(
                        type=FlowRunFilterStateType(
                            any_=["COMPLETED", "FAILED", "CRASHED", "CANCELLED"]
                        )
                    ),
                ),
                sort=FlowRunSort.START_TIME_DESC,
                limit=page_size,
                offset=offset,
            )

            all_flow_runs.extend(flow_runs_page)
            logger.info(f"fetched {len(flow_runs_page)} flow runs (offset={offset})")

            # If we got fewer than page_size, we've reached the end
            if len(flow_runs_page) < page_size:
                break

            offset += page_size

        logger.info(f"found {len(all_flow_runs)} total flow runs")

        # Now filter by AWS environment.
        #
        # Since these are ad-hoc runs, so far, we need to check run
        # parameters or tags The aws_env is likely passed as a
        # parameter to the flow
        filtered_runs = []
        for run in all_flow_runs:
            # Check if parameters contain the AWS env.
            if run.parameters:
                params_aws_env = run.parameters.get("aws_env")
                if params_aws_env and params_aws_env == aws_env.value:
                    filtered_runs.append(run)
                    continue

            # Also check tags
            if run.tags and aws_env.value in [tag.lower() for tag in run.tags]:
                filtered_runs.append(run)
                continue

        logger.info(
            f"filtered to {len(filtered_runs)} flow runs for AWS env {aws_env.value}"
        )

        return filtered_runs


async def collect_vespa_sync_artifacts(
    flow_runs: list[FlowRun],
) -> list[Artifact]:
    """Collect all sync artifacts from the given flow runs."""
    logger = get_logger()

    if not flow_runs:
        logger.info("no flow runs to collect artifacts from")
        return []

    async with get_client() as client:
        # Query for artifacts with key="vespa-sync" from these flow runs with pagination
        all_artifacts = []
        offset = 0
        page_size = 200

        while True:
            artifacts_page = await client.read_artifacts(
                artifact_filter=ArtifactFilter(
                    key=ArtifactFilterKey(any_=["vespa-sync"]),
                    flow_run_id=ArtifactFilterFlowRunId(
                        any_=[str(run.id) for run in flow_runs]
                    ),
                ),
                limit=page_size,
                offset=offset,
            )

            all_artifacts.extend(artifacts_page)
            logger.info(f"fetched {len(artifacts_page)} artifacts (offset={offset})")

            # If we got fewer than page_size, we've reached the end
            if len(artifacts_page) < page_size:
                break

            offset += page_size

        logger.info(f"collected {len(all_artifacts)} vespa-sync artifacts in total")

        return all_artifacts


async def parse_artifacts_to_records(
    artifacts: list[Artifact],
    flow_runs: list[FlowRun],
) -> list[ConceptSyncRecord]:
    """Parse artifacts into structured ConceptSyncRecord objects."""
    logger = get_logger()

    # Create a lookup for flow run metadata
    flow_run_lookup = {str(run.id): run for run in flow_runs}

    records: list[ConceptSyncRecord] = []

    for artifact in artifacts:
        try:
            # Parse the artifact data (it's a JSON string containing a list of dicts)
            if artifact.data is None:
                logger.warning(f"artifact {artifact.id} has no data, skipping")
                continue

            data = json.loads(artifact.data)  # type: ignore[arg-type]

            # Get flow run metadata
            flow_run = flow_run_lookup.get(str(artifact.flow_run_id))
            flow_run_name = flow_run.name if flow_run else "unknown"

            # Parse each row in the artifact table
            for row in data:
                record = ConceptSyncRecord(
                    concept_id=row.get("Concept ID", "Unknown"),
                    wikibase_id=row.get("Wikibase ID", "Unknown"),
                    preferred_label=row.get("Preferred Label", "Unknown"),
                    wikibase_revision=row.get("Wikibase Revision", "N/A"),
                    status=row.get("Status", "Unknown"),
                    error=row.get("Error", "N/A"),
                    flow_run_id=str(artifact.flow_run_id),
                    flow_run_name=flow_run_name,
                    synced_at=artifact.created or datetime.now(timezone.utc),
                )
                records.append(record)

        except Exception as e:
            logger.error(f"failed to parse artifact {artifact.id}: {e}")
            continue

    logger.info(f"parsed {len(records)} concept sync records from artifacts")

    return records


async def merge_concept_records(
    records: list[ConceptSyncRecord],
) -> list[MergedConceptRecord]:
    """
    Merge concept records by Wikibase ID + Concept ID.

    For each unique combination, we keep:
    - The most recent revision and status
    - A count of how many times it was synced
    - All flow runs that touched it
    """
    logger = get_logger()

    # Group by (wikibase_id, concept_id)
    grouped: dict[tuple[str, str], list[ConceptSyncRecord]] = defaultdict(list)

    for record in records:
        key = (record.wikibase_id, record.concept_id)
        grouped[key].append(record)

    # Merge each group
    merged_records: list[MergedConceptRecord] = []

    for (wikibase_id, concept_id), group in grouped.items():
        # Sort by sync time descending (most recent first)
        sorted_group = sorted(group, key=lambda r: r.synced_at, reverse=True)

        # Take latest metadata
        latest = sorted_group[0]

        merged = MergedConceptRecord(
            concept_id=concept_id,
            wikibase_id=wikibase_id,
            preferred_label=latest.preferred_label,
            latest_wikibase_revision=latest.wikibase_revision,
            latest_status=latest.status,
            latest_error=latest.error,
            sync_count=len(group),
            latest_sync_at=latest.synced_at,
            latest_flow_run_name=latest.flow_run_name,
            all_flow_runs=list(set(r.flow_run_name for r in sorted_group)),
        )

        merged_records.append(merged)

    logger.info(
        f"merged {len(records)} records into {len(merged_records)} unique concepts"
    )

    return merged_records


@task(cache_policy=NONE)
async def write_to_google_sheets(
    gcp_credentials_block_name: str,
    spreadsheet_id: str,
    worksheet_name: str,
    merged_records: list[MergedConceptRecord],
) -> None:
    """
    Write merged concept records to a Google Sheets worksheet.

    This task clears the worksheet (excluding the header row) and writes the latest data.
    """
    logger = get_logger()

    logger.info(f"loading GCP credentials block: {gcp_credentials_block_name}")
    gcp_credentials_block = await GcpCredentials.load(gcp_credentials_block_name)

    # Get credentials with Sheets scope
    # service_account_info is a SecretDict, need to get the actual dict value
    service_account_dict = gcp_credentials_block.service_account_info.get_secret_value()
    credentials = service_account.Credentials.from_service_account_info(
        service_account_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )

    # Build the Sheets service
    service = build("sheets", "v4", credentials=credentials)

    logger.info(f"writing {len(merged_records)} records to Google Sheets")

    # Get Wikibase URL from environment or use default
    wikibase_url = os.getenv(
        "WIKIBASE_URL", "https://climatepolicyradar.wikibase.cloud"
    )

    # Prepare the header row
    header_row = [
        "Concept ID",
        "Wikibase ID",
        "Preferred Label",
        "Latest Wikibase Revision",
        "Wikibase Link",
        "Latest Status",
        "Latest Error",
        "Sync Count",
        "Latest Sync",
        "Latest Flow Run Name",
        "All Flow Runs",
    ]

    # Prepare the data rows
    data_rows = [
        [
            record.concept_id,
            record.wikibase_id,
            record.preferred_label,
            str(record.latest_wikibase_revision),
            f"{wikibase_url}/wiki/Item:{record.wikibase_id}",  # Wikibase Link
            record.latest_status,
            record.latest_error,
            record.sync_count,
            record.latest_sync_at.isoformat(),
            record.latest_flow_run_name,
            ", ".join(record.all_flow_runs),
        ]
        for record in merged_records
    ]

    # Clear all existing data (including header)
    try:
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"{worksheet_name}!A1:K")
            .execute()
        )

        existing_rows = len(result.get("values", []))

        if existing_rows > 0:
            logger.info(
                f"clearing {existing_rows} existing rows from worksheet (including header)"
            )
            service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f"{worksheet_name}!A1:K",
            ).execute()
    except Exception as e:
        logger.warning(f"could not check/clear existing data: {e}")

    # Write header and data starting from row 1
    if data_rows:
        # Combine header and data rows
        all_rows = [header_row] + data_rows
        body = {"values": all_rows}

        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f"{worksheet_name}!A1",
            valueInputOption="RAW",
            body=body,
        ).execute()

        logger.info(
            f"successfully wrote header and {len(data_rows)} data rows to {worksheet_name} in spreadsheet {spreadsheet_id}"
        )
    else:
        logger.info("no data to write to Google Sheets")


@flow(persist_result=False)
async def collect_wikibase_vespa_sync_artifacts(
    aws_env: AwsEnv | None = None,
    lookback_days: int = 30,
    gcp_credentials_block_name: str = "jesse-google-prefect",
    google_sheets_spreadsheet_id: str = "1LFWk6Fy_dhAhmJi7aPBWd9FH04IcpGzEr1s9QmUdfP8",
):
    """
    Collect and merge wikibase_to_vespa sync artifacts across multiple flow runs.

    Steps:

    1. Queries Prefect for all wikibase_to_vespa flow runs in the specified AWS env
    2. Collects their vespa-sync artifacts
    3. Parses and merges the concept sync data by Wikibase ID + Concept ID
    4. Creates a summary artifact in Prefect
    5. Writes the data to Google Sheets (staging or production worksheet)
    """
    logger = get_logger()

    # Determine AWS environment
    if aws_env is None:
        aws_env = AwsEnv(os.environ["AWS_ENV"])

    logger.info(
        f"collecting wikibase_to_vespa artifacts for AWS env: {aws_env.value}, "
        f"lookback: {lookback_days} days"
    )

    # Step 1: Get flow runs
    flow_runs = await get_wikibase_to_vespa_flow_runs(
        aws_env=aws_env,
        lookback_days=lookback_days,
    )

    if not flow_runs:
        logger.warning("no flow runs found, nothing to collect")
        return

    # Step 2: Collect artifacts
    artifacts = await collect_vespa_sync_artifacts(flow_runs)

    if not artifacts:
        logger.warning("no artifacts found")
        return

    # Step 3: Parse artifacts to records
    records = await parse_artifacts_to_records(artifacts, flow_runs)

    if not records:
        logger.warning("no records parsed from artifacts")
        return

    # Step 4: Merge records
    merged_records = await merge_concept_records(records)

    # Step 5: Calculate statistics
    successful_syncs = sum(1 for r in records if r.status == "✓")
    failed_syncs = sum(1 for r in records if r.status == "✗")

    logger.info(
        f"statistics: {len(merged_records)} unique concepts, "
        f"{len(records)} total syncs, "
        f"{successful_syncs} successful, "
        f"{failed_syncs} failed"
    )

    # Step 6: Create Prefect artifact with summary
    overview_description = f"""# Wikibase to Vespa Sync Collection Summary

## Overview
- **AWS Environment**: {aws_env.value}
- **Lookback Period**: {lookback_days} days
- **Flow Runs Analyzed**: {len(flow_runs)}
- **Total Sync Events**: {len(records)}
- **Unique Concepts**: {len(merged_records)}
- **Successful Syncs**: {successful_syncs}
- **Failed Syncs**: {failed_syncs}
- **Generated At**: {datetime.now(timezone.utc).isoformat()}

## Notes
This artifact aggregates data from all wikibase_to_vespa flow runs in the specified
environment and time period. Each row represents a unique (Wikibase ID, Concept ID)
combination with the most recent sync status.
"""

    # Get Wikibase URL for artifact links
    wikibase_url = os.getenv(
        "WIKIBASE_URL", "https://climatepolicyradar.wikibase.cloud"
    )

    # Create table data for artifact
    artifact_table = [
        {
            "Concept ID": record.concept_id,
            "Wikibase ID": record.wikibase_id,
            "Preferred Label": record.preferred_label,
            "Latest Revision": str(record.latest_wikibase_revision),
            "Wikibase Link": f"{wikibase_url}/wiki/Item:{record.wikibase_id}",
            "Latest Status": record.latest_status,
            "Latest Error": record.latest_error,
            "Sync Count": record.sync_count,
            "Latest Sync": record.latest_sync_at.isoformat(),
            "Latest Flow Run": record.latest_flow_run_name,
        }
        for record in merged_records
    ]

    await acreate_table_artifact(
        key="wikibase-vespa-sync-collection",
        table=artifact_table,
        description=overview_description,
    )

    logger.info("created Prefect artifact with merged data")

    # Step 7: Write to Google Sheets
    if google_sheets_spreadsheet_id:
        # Determine worksheet name based on environment
        worksheet_name = aws_env.value  # "staging" or "production"

        logger.info(f"writing to Google Sheets worksheet '{worksheet_name}'")

        await write_to_google_sheets(
            gcp_credentials_block_name=gcp_credentials_block_name,
            spreadsheet_id=google_sheets_spreadsheet_id,
            worksheet_name=worksheet_name,
            merged_records=merged_records,
        )
    else:
        logger.info("no Google Sheets spreadsheet ID configured, skipping Sheets write")

    logger.info("flow completed successfully")
