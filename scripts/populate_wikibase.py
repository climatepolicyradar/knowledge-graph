"""
Populates a Wikibase instance with concepts from a JSON file.

Takes the output of the `process_gst.py` script, which contains the processed concepts
hierarchy, and creates a set of corresponding concepts and relationships in Wikibase.

Note: This script assumes that you've already run the `process_gst.py` script to
generate the concepts.json file, and have a Wikibase instance running with the
appropriate credentials set in the .env file.

Usage:
- Run `python scripts/populate_wikibase.py`

Output:
- "data/processed/concepts_with_wikibase_ids.json" containing the original concepts
  hierarchy, along with their new Wikibase IDs.
"""

import json
from logging import getLogger

import dotenv
from tqdm import tqdm

from scripts.config import processed_data_dir
from src.concept import Concept
from src.wikibase import WikibaseSession

logger = getLogger(__name__)

dotenv.load_dotenv()

wikibase = WikibaseSession()


data_path = processed_data_dir / "concepts.json"
with open(data_path, "r") as f:
    concepts_json = json.load(f)

concepts = [Concept.from_dict(concept) for concept in concepts_json]

flat_concepts = [
    concept
    for top_level_concept in concepts
    for concept in top_level_concept.all_subconcepts
] + concepts

concepts_with_wikibase_ids = []
progress_bar = tqdm(total=len(flat_concepts), desc="Creating concepts", unit="concept")
for concept in concepts:
    concept_with_wikibase_id = wikibase.create_concept(
        concept, progress_bar=progress_bar
    )
    concepts_with_wikibase_ids.append(concept_with_wikibase_id)
progress_bar.close()

with open("./data/processed/concepts_with_wikibase_ids.json", "w") as f:
    json.dump([concept.dict() for concept in concepts_with_wikibase_ids], f)
