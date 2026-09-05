from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def make_base_estimator(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "linear_svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            dual="auto",
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def train_model(model_name: str, X_train, y_train):
    estimator = make_base_estimator(model_name)
    model = MultiOutputClassifier(estimator, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
