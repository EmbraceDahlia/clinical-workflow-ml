from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMNS


# def select_feature_columns(df: pd.DataFrame) -> list[str]:
#     excluded = {
#         "encounter_id", "patient_id", "INDEX_DATE", 
#     }
#     excluded |= set(TARGET_COLUMNS)
#     return [column for column in df.columns if column not in excluded]

def select_feature_columns(
    df: pd.DataFrame,
    experiment: str = "full",
) -> list[str]:

    excluded = {
        "encounter_id",
        "patient_id",
        "INDEX_DATE",
    }

    excluded |= set(TARGET_COLUMNS)

    # Experiment 1: basic demographic + current encounter information
    basic_features = {
        "GENDER",
        "age_at_encounter",
        "ENCOUNTERCLASS",
        "encounter_code",
        "reason_code",
    }

    # Longitudinal EHR history
    longitudinal_features = {
        "patient_total_encounters",
        "days_since_last_encounter",
        "days_since_allergies_last",
        "days_since_conditions_last",
        "days_since_immunizations_last",
        "days_since_procedures_last",
        "days_since_imaging_last",
        "days_since_medications_last",
        "days_since_observations_last",
    }

    # Add all prior_* count features
    longitudinal_features |= {
        column
        for column in df.columns
        if column.startswith("prior_")
    }

    # Prior screening measurements
    screening_features = {
        column
        for column in df.columns
        if column.startswith("screening_prior_")
    }

    if experiment == "basic":
        allowed_features = basic_features

    elif experiment == "longitudinal":
        allowed_features = (
            basic_features
            | longitudinal_features
        )

    elif experiment == "full":
        allowed_features = (
            basic_features
            | longitudinal_features
            | screening_features
        )

    else:
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            "Use 'basic', 'longitudinal', or 'full'."
        )

    feature_columns = [
        column
        for column in df.columns
        if column in allowed_features
        and column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            f"No features selected for experiment '{experiment}'"
        )

    return feature_columns


def build_preprocessor(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> ColumnTransformer:
    # feature_columns = select_feature_columns(df,experiment="full")
    if not feature_columns:
        raise ValueError("No usable feature columns found in the modeling table")

    numeric_cols = df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [column for column in feature_columns if column not in numeric_cols]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No numeric or categorical transformer configuration could be built")

    return ColumnTransformer(transformers=transformers, remainder="drop")

