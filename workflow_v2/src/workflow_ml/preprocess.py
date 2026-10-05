"""Dense, scaled inputs for the neural models (trees use the raw table directly)."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from workflow_ml.features import CATEGORICAL_COLUMNS


def signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def dense_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in columns if column not in CATEGORICAL_COLUMNS]
    # Day counts are heavy-tailed, so compress them; "never happened" becomes a flag.
    numeric_steps = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
        FunctionTransformer(signed_log),
        StandardScaler(),
    )
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                            sparse_output=False, dtype=np.float32)
    return ColumnTransformer([("numeric", numeric_steps, numeric), ("categorical", one_hot, categorical)])
