import numpy as np
import pandas as pd
import pytest

from core.errors import ServiceError
from ml.dataset import load_dataset, prepare_data, split_dataset, validate_dataset
from ml.pipeline import train_churn_model
from schemas.churn import FEATURE_NAMES, NUMERIC_FEATURES, TrainingConfigChurn


def test_provided_csv_and_stratified_split(dataset):
    assert dataset.shape == (2000, 10)
    assert dataset.churn.value_counts().to_dict() == {0: 1597, 1: 403}
    first = split_dataset(dataset)
    second = split_dataset(dataset)
    assert first.info() == {
        "train_rows": 1600,
        "test_rows": 400,
        "test_size": 0.2,
        "random_state": 42,
        "train_class_distribution": {"0": 1278, "1": 322},
        "test_class_distribution": {"0": 319, "1": 81},
    }
    assert set(first.train.index).isdisjoint(first.test.index)
    assert len(set(first.train.index) | set(first.test.index)) == len(dataset)
    assert first.train.index.tolist() == second.train.index.tolist()


def test_prepare_data_orders_columns_and_separates_target(dataset):
    features, labels = prepare_data(dataset[list(reversed(dataset.columns))])
    assert list(features.columns) == FEATURE_NAMES
    assert "churn" not in features
    assert labels.tolist() == dataset.churn.tolist()


def test_missing_values_fitted_only_on_training_data(dataset):
    train = split_dataset(dataset).train.copy()
    train.loc[train.index[:20], "usage_hours"] = np.nan
    train.loc[train.index[:20], "region"] = None
    pipeline = train_churn_model(train)
    numeric = pipeline["preprocessor"].named_transformers_["numeric"]
    median = numeric["imputer"].statistics_[NUMERIC_FEATURES.index("usage_hours")]
    assert median == pytest.approx(train.usage_hours.median())
    # Приём новых значений не переобучает медианы и масштабирование.
    before = numeric["scaler"].mean_.copy()
    features, _ = prepare_data(dataset.head(3))
    features.loc[:, "usage_hours"] = 1000000.0
    assert np.isfinite(pipeline.predict_proba(features)).all()
    np.testing.assert_array_equal(numeric["scaler"].mean_, before)


def test_all_missing_column_and_unseen_supported_category(dataset):
    train = dataset[dataset.region != "africa"].head(300).copy()
    train.loc[:, "usage_hours"] = np.nan
    pipeline = train_churn_model(train)
    sample = dataset[dataset.region == "africa"].head(2)
    features, _ = prepare_data(sample)
    probabilities = pipeline.predict_proba(features)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "kind", ["empty", "missing_column", "bad_label", "bad_category"]
)
def test_invalid_datasets_rejected(dataset, kind):
    frame = dataset.head(20).copy()
    if kind == "empty":
        frame = frame.head(0)
    elif kind == "missing_column":
        frame = frame.drop(columns="monthly_fee")
    elif kind == "bad_label":
        frame.loc[frame.index[0], "churn"] = 3
    else:
        frame.loc[frame.index[0], "region"] = "unknown"
    with pytest.raises(ServiceError):
        validate_dataset(frame)


def test_one_class_cannot_be_split(dataset):
    with pytest.raises(ServiceError, match="оба класса"):
        split_dataset(dataset[dataset.churn == 0])


@pytest.mark.parametrize("content", ["", "monthly_fee,churn\n", "bad,columns\n1,0\n"])
def test_invalid_csv(tmp_path, content):
    path = tmp_path / "input.csv"
    path.write_text(content)
    with pytest.raises(ServiceError):
        load_dataset(path)


def test_csv_missing_features_are_represented_as_none(dataset, tmp_path):
    frame = dataset.head(30).copy()
    frame.loc[frame.index[0], "region"] = None
    frame.loc[frame.index[1], "support_requests"] = np.nan
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert pd.isna(loaded.iloc[0].region)
    assert pd.isna(loaded.iloc[1].support_requests)
    assert loaded.churn.notna().all()


@pytest.mark.parametrize("model_type", ["logreg", "random_forest"])
def test_training_reproducible(dataset, model_type):
    config = TrainingConfigChurn(
        model_type=model_type,
        hyperparameters={"n_estimators": 10} if model_type == "random_forest" else {},
    )
    split = split_dataset(dataset)
    first = train_churn_model(split.train, config)
    second = train_churn_model(split.train, config)
    features, _ = prepare_data(split.test)
    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features)
    )
