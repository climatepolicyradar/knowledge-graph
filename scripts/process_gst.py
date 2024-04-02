import itertools
import json
import warnings
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import pandas as pd
from explorer.main import load_input_spreadsheet

from src.concept import Concept

logger = getLogger(__name__)

# ignore UserWarnings from openpyxl about data validation
warnings.simplefilter(action="ignore", category=UserWarning)

# assume that the global-stocktake repo is cloned in the same directory as the
# knowledge-graph repo
assert Path(
    "../global-stocktake"
).exists(), "Please clone the global-stocktake repo in the same directory as the knowledge-graph repo"

# set up the input and output directories
input_data_dir = Path("../global-stocktake/concepts")
output_data_dir = Path("./data/processed")
output_data_dir.mkdir(parents=True, exist_ok=True)

concepts = []

for concept_dir in input_data_dir.iterdir():
    if concept_dir.is_dir():
        file = concept_dir / "input.xlsx"
        if not file.exists():
            logger.info(f"Skipping {concept_dir.stem} as no input.xlsx file found")
            continue

        logger.info(f"Processing {concept_dir.stem}")

        # load all the synonyms for each span id
        level_1_concept = Concept(
            preferred_label=concept_dir.stem.replace("-", " ")
        )  # the directory name

        patterns, _ = load_input_spreadsheet(concept_dir / "input.xlsx")
        patterns = sorted(patterns, key=lambda i: i.get("id", ""))

        all_synonyms = defaultdict(list)

        for span_id, rules in itertools.groupby(patterns, lambda i: i.get("id", "")):
            patterns = [p["pattern"] for p in rules]

            for pattern in patterns:
                tokens = []

                for token in pattern:
                    token_val = list(token.values())[0]

                    if isinstance(token_val, str):
                        tokens.append([token_val])
                    elif isinstance(token_val, dict):
                        token_vals_list = list(token_val.values())[0]
                        tokens.append(list(set([i.lower() for i in token_vals_list])))

                    else:
                        logger.warning(
                            f"Unknown token type: {type(token_val)} for {token_val}"
                        )

                all_synonyms[span_id] += list(itertools.product(*tokens))

        # join up the individual token lists so that each synonym is just represented by a single string
        # and then replace any instances of " - " with "-"
        all_synonyms = {
            k: [" ".join(i).replace(" -", "-").replace("- ", "-") for i in list(set(v))]
            for k, v in all_synonyms.items()
        }

        # now build the concept hierarchy
        df = pd.read_excel(file)
        for _, row in df.dropna(subset=["Span label", "Span ID (optional)"]).iterrows():
            # if row["Span label"] is already in the hierarchy, use it as the level 2 concept
            # otherwise, create a new level 2 concept
            if row["Span label"] in [
                c.preferred_label for c in level_1_concept.subconcepts
            ]:
                level_2_concept = level_1_concept[row["Span label"]]
            else:
                level_2_concept = Concept(preferred_label=row["Span label"])
                level_1_concept.subconcepts.append(level_2_concept)

            concept_synonyms = all_synonyms.get(
                row["Span ID (optional)"].upper().strip(), []
            )

            # if row["Span ID (optional)"] is already in the hierarchy, use it as the level 3 concept
            # otherwise, create a new level 3 concept
            if row["Span ID (optional)"] in [
                c.preferred_label for c in level_2_concept.subconcepts
            ]:
                level_3_concept = level_2_concept[row["Span ID (optional)"]]
                level_3_concept.alternative_labels.update(concept_synonyms)
            else:
                level_3_concept = Concept(
                    preferred_label=row["Span ID (optional)"],
                    alternative_labels=concept_synonyms,
                )
                level_2_concept.subconcepts.append(level_3_concept)

        concepts.append(level_1_concept)

# dump the data to a json file
json_output_path = output_data_dir / "concepts.json"
with open(json_output_path, "w") as f:
    json.dump([c.dict() for c in concepts], f, indent=2)

logger.info(f"Wrote concepts to {json_output_path}")
