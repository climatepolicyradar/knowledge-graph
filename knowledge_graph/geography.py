import pandas as pd
from rapidfuzz import fuzz

from knowledge_graph.config import raw_data_dir

geography_data = raw_data_dir / "geography-iso-3166-backend.csv"
geography_df = pd.read_csv(geography_data)


def use_cache(func):
    CACHE = {}

    def wrapper(*args, **kwargs):
        if args[0] in CACHE:
            return CACHE[args[0]]
        result = func(*args, **kwargs)
        CACHE[args[0]] = result
        return result

    return wrapper


@use_cache
def geography_string_to_iso(geography_string: str) -> str:
    """Run a fuzzy search on the geography string to find the best match in the geography data"""
    best_match = ""
    best_score = 0
    for _, row in geography_df.iterrows():
        score = fuzz.ratio(geography_string, row["Name"])  # type: ignore
        if score > best_score:
            best_score = score
            best_match = row["Iso"]
    return best_match  # type: ignore


iso_to_world_bank_region = geography_df.set_index("Iso")["World Bank Region"].to_dict()
