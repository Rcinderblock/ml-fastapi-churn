"""Обучаемая предобработка и классификатор сохраняются одним Pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import prepare_data
from schemas.churn import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MetricsChurn,
    TrainingConfigChurn,
)


def build_pipeline(config: TrainingConfigChurn) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="missing", keep_empty_features=True
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    if config.model_type == "logreg":
        classifier = LogisticRegression(**config.hyperparameters)
    else:
        classifier = RandomForestClassifier(**config.hyperparameters, n_jobs=1)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_churn_model(
    dataset: pd.DataFrame, config: TrainingConfigChurn | None = None
) -> Pipeline:
    """Обучить всю цепочку только на переданной тренировочной выборке."""
    features, labels = prepare_data(dataset)
    pipeline = build_pipeline(config or TrainingConfigChurn())
    return pipeline.fit(features, labels)


def evaluate_churn_model(pipeline: Pipeline, test: pd.DataFrame) -> MetricsChurn:
    features, labels = prepare_data(test)
    predictions = pipeline.predict(features)
    positive_index = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(features)[:, positive_index]
    return MetricsChurn(
        accuracy=float(accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    )
