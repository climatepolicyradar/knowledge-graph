# Geography Concepts

This directory contains the scripts which were used to populate the concept store with an initial set of geography concepts. The process was as follows:

1. Domain experts were consulted to identify the most relevant taxonomy of geography concepts, selecting ISO 3166-1 as the most useful.
2. [A script](01_copy_geographies_from_wikidata.py) was written to copy the ISO 3166-1 geography concepts from Wikidata to a local CSV file.
3. These concepts were then manually reviewed and their aliases were tweaked to most useful for searching and disambiguation. The refined sheet is in [google sheets, here](https://docs.google.com/spreadsheets/d/1f6YupXPK7DGcwXEnqCRoBHjumkg-pY0unj0S2_CRQPs/edit?usp=sharing). To get a local copy of this dataset for use in the second script, select the "ISO 3166-1 aliases_cleaned" sheet, and download it as a CSV via **File > Download > Comma Separated Values (.csv)**. Save it to the `scripts/geography_concepts/data` directory as `iso_3166-1_aliases_cleaned.csv`.
4. [A second script](02_upload_geographies_to_wikibase.py) was written to upload the refined data to the concept store, preserving the hierarchy of the root 'geography' concept → region-level concepts → national-level concepts.

This directory is maintained solely for archival purposes - the scripts here aren't maintained, and aren't intended to be re-run. Data in the concept store has likely changed since these scripts were first run, and thus the scripts here are not guaranteed to work.

## A note on unused data

The datasets fetched from Wikidata by [01_copy_geographies_from_wikidata.py](01_copy_geographies_from_wikidata.py) include subnational and historical geography concepts, which were not used in the final dataset. While the national-level concepts produce a list of ~250 concepts to add to the concept store, adding the subnational and historical concepts would have added an extra ~5,000 concepts (tripling the size of the existing concept store), which we deemed unmanageable and unnecessary for this first iteration. However, we could still add them in the future if/when they're needed.

The audited and approved dataset from google sheets contains a column for 'region', which was included in the uploaded taxonomy, but the 'sub-region' column was not. Again, this choice was made because we deemed the sub-region hierarchy to be too granular for the first iteration. We'll wait for more feedback on how the programmes team want to use this dataset before deciding on which new hierarchy levels to add to the geography taxonomy, eg they may instead prefer to add groups for Least Developed Countries, Small Island Developing States, World Bank income groupings etc instead of ISO sub-regions.

See more discussion in the [#proj-geographies](https://climate-policy-radar.slack.com/archives/C0AKU9Y6WF7) channel on slack.
