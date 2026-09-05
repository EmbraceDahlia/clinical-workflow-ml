from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.procedure_mapping import build_procedure_mapping


DATA_PATH = ROOT / "data" / "procedures.csv"
OUTPUT_PATH = ROOT / "procedure_category_mapping.csv"


def main() -> None:
    procedures = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    mapping = build_procedure_mapping(procedures)
    mapping.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(mapping):,} unique procedure mappings to {OUTPUT_PATH.name}")
    print(mapping["CATEGORY"].value_counts().to_string())


if __name__ == "__main__":
    main()
