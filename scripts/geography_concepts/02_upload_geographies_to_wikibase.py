"""
Upload ISO 3166-1 national geography concepts to Wikibase.

Creates a two-level hierarchy under the geography root concept (Q2032):
  geography → region → country

Countries with no region in the CSV are attached directly to Q2032.

Usage:
    python scripts/geography_concepts/02_upload_geographies_to_wikibase.py
    python scripts/geography_concepts/02_upload_geographies_to_wikibase.py --dry-run
    python scripts/geography_concepts/02_upload_geographies_to_wikibase.py path/to/other.csv
"""

import csv
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession

GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID = WikibaseID("Q2032")
DEFAULT_CSV = Path(__file__).parent / "data" / "iso_3166-1_aliases_cleaned.csv"

console = Console()
app = typer.Typer()


def build_country_concept(row: dict) -> Concept:
    """Build a Concept from a CSV row, collecting all available alternative labels."""
    preferred_label = row["ISO short name"].strip()

    alt_labels: set[str] = set()
    if alpha2 := row.get("alpha-2", "").strip():
        alt_labels.add(alpha2)
    if alpha3 := row.get("alpha-3", "").strip():
        alt_labels.add(alpha3)

    for alias in row.get("wikidata_alternative_labels", "").split("|"):
        if alias := alias.strip():
            alt_labels.add(alias)

    return Concept(
        preferred_label=preferred_label,
        alternative_labels=sorted(alt_labels),
    )


@app.command()
def upload(
    csv_path: Annotated[
        Path, typer.Argument(help="Path to the cleaned ISO 3166-1 CSV")
    ] = DEFAULT_CSV,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview what would be created without writing"),
    ] = False,
) -> None:
    """Upload ISO 3166-1 geography concepts to Wikibase."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    console.log(f"Loaded {len(rows)} rows from {csv_path}")

    # Collect unique region names in first-appearance order
    region_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        parent_region = row["region"].strip()
        if parent_region and parent_region not in seen:
            region_names.append(parent_region)
            seen.add(parent_region)

    if dry_run:
        console.log(
            f"[dry-run] Would create {len(region_names)} region concepts under {GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID}:"
        )
        for name in region_names:
            console.log(f"  {name!r}")
        console.log(f"[dry-run] Would create {len(rows)} country concepts:")
        for row in rows:
            concept_data = build_country_concept(row)
            parent_region = row["region"].strip() or str(
                GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID
            )
            console.log(f"  {concept_data.preferred_label!r} under {parent_region!r}")
        return

    wikibase = WikibaseSession()

    # Phase 1: create region concepts as subconcepts of Q2032
    console.log(f"Creating {len(region_names)} region concepts...")
    region_ids: dict[str, WikibaseID] = {}
    for name in region_names:
        region_id = wikibase.create_concept(
            Concept(preferred_label=name),
            subconcept_of=[GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID],
        )
        wikibase.add_claim(
            entity_id=GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID,
            property_id=wikibase.has_subconcept_property_id,
            target_id=region_id,
        )
        region_ids[name] = region_id
        console.log(f"  {name!r} -> {region_id}")

    # Phase 2: create country concepts as subconcepts of their region (or Q2032)
    console.log(f"Creating {len(rows)} country concepts...")
    for row in rows:
        concept_data = build_country_concept(row)
        parent_region = row["region"].strip()
        parent_region_id = region_ids.get(
            parent_region, GEOGRAPHY_ROOT_CONCEPT_WIKIBASE_ID
        )
        country_id = wikibase.create_concept(
            concept_data,
            subconcept_of=[parent_region_id],
            wikidata_id=row["wikidata_id"].strip() or None,
        )
        wikibase.add_claim(
            entity_id=parent_region_id,
            property_id=wikibase.has_subconcept_property_id,
            target_id=country_id,
        )
        console.log(f"  {concept_data.preferred_label!r} -> {country_id}")
    console.log("Done.")


if __name__ == "__main__":
    app()
