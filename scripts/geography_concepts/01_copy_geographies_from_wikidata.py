"""
Fetch geography concepts from Wikidata and save them as JSON or CSV files.

See the RFC for full context:
https://www.notion.so/3119109609a480fd884ededd6b1af8aa

Three subcommands cover the three ISO 3166 levels:

  national     — ISO 3166-1, e.g. France, United States
  subnational  — ISO 3166-2, e.g. California (US-CA), nested under their parent country
  historical   — ISO 3166-3, e.g. Czechoslovakia (CSHH)

Output filenames are fixed per subcommand (iso_3166-1, iso_3166-2, iso_3166-3).
Use --format to choose json (default) or csv.

Usage:
    python scripts/geography_concepts/01_copy_geographies_from_wikidata.py national --format csv
    python scripts/geography_concepts/01_copy_geographies_from_wikidata.py subnational --format csv
    python scripts/geography_concepts/01_copy_geographies_from_wikidata.py historical --format csv
"""

import csv
import enum
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from knowledge_graph.wikidata import WikidataSession

# Wikidata property IDs
ISO_3166_ALPHA2_PROPERTY = "P297"
ISO_3166_ALPHA3_PROPERTY = "P298"
ISO_3166_2_PROPERTY = "P300"
ISO_3166_3_PROPERTY = "P773"

COUNTRIES_SPARQL_QUERY = f"""
SELECT DISTINCT ?country ?countryLabel ?countryDescription
       ?{ISO_3166_ALPHA2_PROPERTY} ?{ISO_3166_ALPHA3_PROPERTY}
       (GROUP_CONCAT(DISTINCT ?enAlias; separator=" | ") AS ?enAliases)
       (GROUP_CONCAT(DISTINCT ?mulAlias; separator=" | ") AS ?mulAliases)
WHERE {{
  ?country wdt:{ISO_3166_ALPHA2_PROPERTY} ?{ISO_3166_ALPHA2_PROPERTY} .
  OPTIONAL {{ ?country wdt:{ISO_3166_ALPHA3_PROPERTY} ?{ISO_3166_ALPHA3_PROPERTY} }}
  OPTIONAL {{ ?country skos:altLabel ?enAlias . FILTER(LANG(?enAlias) = "en") }}
  OPTIONAL {{ ?country skos:altLabel ?mulAlias . FILTER(LANG(?mulAlias) = "mul") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
GROUP BY ?country ?countryLabel ?countryDescription ?{ISO_3166_ALPHA2_PROPERTY} ?{ISO_3166_ALPHA3_PROPERTY}
ORDER BY ?{ISO_3166_ALPHA2_PROPERTY}
"""

SUBNATIONAL_SPARQL_QUERY = f"""
SELECT DISTINCT ?subdivision ?subdivisionLabel ?subdivisionDescription
       ?{ISO_3166_2_PROPERTY} ?parentCountry
       (GROUP_CONCAT(DISTINCT ?enAlias; separator=" | ") AS ?enAliases)
       (GROUP_CONCAT(DISTINCT ?mulAlias; separator=" | ") AS ?mulAliases)
WHERE {{
  ?subdivision wdt:{ISO_3166_2_PROPERTY} ?{ISO_3166_2_PROPERTY} .
  BIND(STRBEFORE(?{ISO_3166_2_PROPERTY}, "-") AS ?alpha2)
  OPTIONAL {{ ?parentCountry wdt:{ISO_3166_ALPHA2_PROPERTY} ?alpha2 }}
  OPTIONAL {{ ?subdivision skos:altLabel ?enAlias . FILTER(LANG(?enAlias) = "en") }}
  OPTIONAL {{ ?subdivision skos:altLabel ?mulAlias . FILTER(LANG(?mulAlias) = "mul") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
GROUP BY ?subdivision ?subdivisionLabel ?subdivisionDescription ?{ISO_3166_2_PROPERTY} ?parentCountry
ORDER BY ?{ISO_3166_2_PROPERTY}
"""

HISTORICAL_SPARQL_QUERY = f"""
SELECT DISTINCT ?country ?countryLabel ?countryDescription
       ?{ISO_3166_3_PROPERTY}
       (GROUP_CONCAT(DISTINCT ?enAlias; separator=" | ") AS ?enAliases)
       (GROUP_CONCAT(DISTINCT ?mulAlias; separator=" | ") AS ?mulAliases)
WHERE {{
  ?country wdt:{ISO_3166_3_PROPERTY} ?{ISO_3166_3_PROPERTY} .
  OPTIONAL {{ ?country skos:altLabel ?enAlias . FILTER(LANG(?enAlias) = "en") }}
  OPTIONAL {{ ?country skos:altLabel ?mulAlias . FILTER(LANG(?mulAlias) = "mul") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
GROUP BY ?country ?countryLabel ?countryDescription ?{ISO_3166_3_PROPERTY}
ORDER BY ?{ISO_3166_3_PROPERTY}
"""

console = Console()
app = typer.Typer()


class Format(str, enum.Enum):
    """The format to write the output in."""

    json = "json"
    csv = "csv"


def binding_value(binding: dict[str, dict], key: str) -> Optional[str]:
    """Extract a string value from a SPARQL result binding, or None if absent."""
    return binding.get(key, {}).get("value")


def parse_aliases(binding: dict[str, dict]) -> list[str]:
    """Extract and split English and multilingual aliases from a binding."""
    aliases: list[str] = []
    for key in ("enAliases", "mulAliases"):
        raw = binding_value(binding, key) or ""
        aliases.extend(a.strip() for a in raw.split(" | ") if a.strip())
    return aliases


DATA_DIR = Path(__file__).parent / "data"


def write_output(geographies: list[dict], stem: str, fmt: Format) -> None:
    """Write geographies to a file with a fixed name derived from the ISO standard."""
    path = DATA_DIR / f"{stem}.{fmt.value}"
    if fmt == Format.json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(geographies, f, indent=2, ensure_ascii=False)
    else:
        fieldnames = list(geographies[0].keys()) if geographies else []
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for geo in geographies:
                writer.writerow(
                    {
                        k: " | ".join(v) if isinstance(v, list) else (v or "")
                        for k, v in geo.items()
                    }
                )
    console.log(f"Saved {len(geographies)} concepts to {path}")


@app.command()
def national(
    fmt: Annotated[Format, typer.Option("--format")] = Format.json,
) -> None:
    """Fetch ISO 3166-1 national geography concepts from Wikidata."""
    wikidata = WikidataSession()
    console.log("Fetching national concepts from Wikidata...")
    bindings = wikidata.run_sparql_query(COUNTRIES_SPARQL_QUERY)

    geographies = []
    for binding in bindings:
        qid_uri = binding_value(binding, "country")
        if not qid_uri:
            continue

        alt_labels: list[str] = []
        for prop in (ISO_3166_ALPHA2_PROPERTY, ISO_3166_ALPHA3_PROPERTY):
            if code := binding_value(binding, prop):
                alt_labels.append(code)
        alt_labels.extend(parse_aliases(binding))

        geographies.append(
            {
                "wikidata_id": qid_uri.split("/")[-1],
                "preferred_label": binding_value(binding, "countryLabel") or "Unknown",
                "description": binding_value(binding, "countryDescription"),
                "alternative_labels": sorted(set(alt_labels)),
            }
        )

    console.log(f"Parsed {len(geographies)} national concepts")

    table = Table(
        "Wikidata ID", "Preferred label", "ISO alpha-2 / alpha-3", "# alt labels"
    )
    for geo in geographies:
        iso_codes = [
            label
            for label in geo["alternative_labels"]
            if len(label) in (2, 3) and label.isupper()
        ]
        table.add_row(
            geo["wikidata_id"],
            geo["preferred_label"],
            " / ".join(iso_codes),
            str(len(geo["alternative_labels"])),
        )
    console.print(table)
    write_output(geographies, "iso_3166-1", fmt)


@app.command()
def subnational(
    fmt: Annotated[Format, typer.Option("--format")] = Format.json,
) -> None:
    """Fetch ISO 3166-2 subnational geography concepts from Wikidata, nested under their parent country."""
    wikidata = WikidataSession()
    console.log("Fetching subnational concepts from Wikidata...")
    bindings = wikidata.run_sparql_query(SUBNATIONAL_SPARQL_QUERY)

    geographies = []
    unresolved = 0
    for binding in bindings:
        qid_uri = binding_value(binding, "subdivision")
        if not qid_uri:
            continue

        alt_labels: list[str] = []

        if iso2 := binding_value(binding, ISO_3166_2_PROPERTY):
            alt_labels.append(iso2)
        alt_labels.extend(parse_aliases(binding))

        parent_uri = binding_value(binding, "parentCountry")
        parent_wikidata_id = parent_uri.split("/")[-1] if parent_uri else None
        if parent_wikidata_id is None:
            unresolved += 1

        geographies.append(
            {
                "wikidata_id": qid_uri.split("/")[-1],
                "preferred_label": binding_value(binding, "subdivisionLabel")
                or "Unknown",
                "description": binding_value(binding, "subdivisionDescription"),
                "parent_wikidata_id": parent_wikidata_id,
                "alternative_labels": sorted(set(alt_labels)),
            }
        )

    console.log(
        f"Parsed {len(geographies)} subnational concepts ({unresolved} with no parent resolved)"
    )

    table = Table("Wikidata ID", "Preferred label", "ISO 3166-2", "Parent Wikidata ID")
    for geo in geographies:
        iso_code = next(
            (
                label
                for label in geo["alternative_labels"]
                if "-" in label and label.isupper()
            ),
            "",
        )
        table.add_row(
            geo["wikidata_id"],
            geo["preferred_label"],
            iso_code,
            geo["parent_wikidata_id"] or "[unresolved]",
        )
    console.print(table)
    write_output(geographies, "iso_3166-2", fmt)


@app.command()
def historical(
    fmt: Annotated[Format, typer.Option("--format")] = Format.json,
) -> None:
    """
    Fetch ISO 3166-3 historical geography concepts from Wikidata.

    ISO 3166-3 is the subdivision of ISO 3166 which denotes "Code for formerly used
    names of countries." See more here: https://en.wikipedia.org/wiki/ISO_3166-3
    """
    wikidata = WikidataSession()
    console.log("Fetching historical geography concepts from Wikidata...")
    bindings = wikidata.run_sparql_query(HISTORICAL_SPARQL_QUERY)

    geographies = []
    for binding in bindings:
        qid_uri = binding_value(binding, "country")
        if not qid_uri:
            continue

        alt_labels: list[str] = []
        if iso3 := binding_value(binding, ISO_3166_3_PROPERTY):
            alt_labels.append(iso3)
        alt_labels.extend(parse_aliases(binding))

        geographies.append(
            {
                "wikidata_id": qid_uri.split("/")[-1],
                "preferred_label": binding_value(binding, "countryLabel") or "Unknown",
                "description": binding_value(binding, "countryDescription"),
                "alternative_labels": sorted(set(alt_labels)),
            }
        )

    console.log(f"Parsed {len(geographies)} historical geography concepts")

    table = Table("Wikidata ID", "Preferred label", "ISO 3166-3", "# alt labels")
    for geo in geographies:
        iso_code = next(
            (
                label
                for label in geo["alternative_labels"]
                if label.isupper() and len(label) == 4
            ),
            "",
        )
        table.add_row(
            geo["wikidata_id"],
            geo["preferred_label"],
            iso_code,
            str(len(geo["alternative_labels"])),
        )
    console.print(table)
    write_output(geographies, "iso_3166-3", fmt)


if __name__ == "__main__":
    app()
