"""Patient-state phenotypes: PCA compresses the features, K-Means groups similar encounters."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from workflow_ml.config import SEED
from workflow_ml.features import feature_columns
from workflow_ml.preprocess import dense_preprocessor


class PhenotypeEncoder:
    """Adds the nearest cluster and the distance to every cluster centre as extra features."""

    def __init__(self, components: int = 20, clusters: int = 16, seed: int = SEED):
        self.components, self.clusters, self.seed = components, clusters, seed

    def fit(self, table: pd.DataFrame, rows: np.ndarray) -> "PhenotypeEncoder":
        self.columns = feature_columns(table)
        self.pipeline = make_pipeline(
            dense_preprocessor(self.columns),
            PCA(self.components, random_state=self.seed),
            KMeans(self.clusters, n_init=5, random_state=self.seed),
        )
        self.pipeline.fit(table.iloc[rows][self.columns])
        return self

    def transform(self, table: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        distances = self.pipeline.transform(table.iloc[rows][self.columns])
        names = [f"phenotype_distance_{cluster}" for cluster in range(self.clusters)]
        frame = pd.DataFrame(distances, columns=names, index=table.index[rows])
        frame["phenotype"] = distances.argmin(axis=1)
        return frame
