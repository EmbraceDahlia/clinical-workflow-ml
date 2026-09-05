
from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_TABLE_PATH, TARGET_COLUMNS, TIME_WINDOWS
from src.procedure_mapping import classify_procedure


DATA = ROOT / "data"
OUTPUT = MODEL_TABLE_PATH
HORIZON_DAYS = 30


# print("\nTarget distribution:")
# print("Number of rows:", len(df))
# print("\nCount of positive labels:")
# print(df[TARGET_COLUMNS].sum())

# print("\nPercentage of positive labels:")
# print((df[TARGET_COLUMNS].mean() * 100).round(2))

# print("\nNumber of labels per encounter:")
# print(df[TARGET_COLUMNS].sum(axis=1).value_counts().sort_index())


# procedures = pd.read_csv(DATA / "procedures.csv", encoding="utf-8-sig", low_memory=False)

# classified = procedures["DESCRIPTION"].apply(classify_procedure)

# procedures["CATEGORY"] = classified.str[0]

# print(
#     procedures.loc[
#         procedures["CATEGORY"] == "instrument_test",
#         "DESCRIPTION"
#     ].value_counts().head(30)
# )

encounters = pd.read_csv(
    DATA / "encounters.csv",
    encoding="utf-8-sig",
    low_memory=False,
)

print("--- Encounter field inspection ---")

print("Total encounters:", len(encounters))
print("Unique CODE:", encounters["CODE"].nunique())
print("Unique ENCOUNTERCLASS:", encounters["ENCOUNTERCLASS"].nunique())
print("Unique REASONCODE:", encounters["REASONCODE"].nunique())

print("\n--- Encounter classes ---")
print(encounters["ENCOUNTERCLASS"].value_counts())

print("\n--- REASONCODE missing percentage ---")
print(round(encounters["REASONCODE"].isna().mean() * 100, 2))

print("\n--- Most common REASONDESCRIPTION ---")
print(encounters["REASONDESCRIPTION"].value_counts().head(10))

print("\n--- CODE → DESCRIPTION relationship ---")
code_description_counts = (
    encounters.groupby("CODE")["DESCRIPTION"]
    .nunique()
)

print(code_description_counts.value_counts().sort_index())