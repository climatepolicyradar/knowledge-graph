# Geography Concepts

This directory contains the scripts which were used to populate the concept store with an initial set of geography concepts. The process was as follows:

1. Domain experts were consulted to identify the most relevant taxonomy of geography concepts, selecting ISO 3166-1 as the most useful.
2. [A script](01_copy_geographies_from_wikidata.py) was written to copy the ISO 3166-1 geography concepts from Wikidata to a local CSV file.
3. These concepts were then manually reviewed and their aliases were tweaked to most useful for searching and disambiguation. The refined sheet is in [google sheets, here](https://docs.google.com/spreadsheets/d/1f6YupXPK7DGcwXEnqCRoBHjumkg-pY0unj0S2_CRQPs/edit?usp=sharing). To get a local copy of this dataset for use in the second script, select the "ISO 3166-1 aliases_cleaned" sheet, and download it as a CSV via **File > Download > Comma Separated Values (.csv)**. Save it to the `scripts/geography_concepts/data` directory as `iso_3166-1_aliases_cleaned.csv`.
4. [A second script](02_upload_geographies_to_wikibase.py) was written to upload the refined data to the concept store, preserving the hierarchy of the root 'geography' concept → region-level concepts → national-level concepts.

This directory is maintained solely for archival purposes - the scripts here aren't maintained, and aren't intended to be re-run. Data in the concept store has likely changed since these scripts were first run, and thus the scripts here are not guaranteed to work.
