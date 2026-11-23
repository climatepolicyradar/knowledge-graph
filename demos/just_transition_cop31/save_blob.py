"""
Turn a pipe-separated CSV blob copied from the Snowflake MCP into data/<name>.csv.

Usage: python save_blob.py <name> < blob.txt
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

name = sys.argv[1]
blob = sys.stdin.read().strip()
df = pd.read_csv(StringIO(blob.replace("|", "\n")))
out = Path(__file__).parent / "data" / f"{name}.csv"
df.to_csv(out, index=False)
print(f"{out.name}: {len(df)} rows, columns {list(df.columns)}")
