import pytest
from fastapi.testclient import TestClient

from core.config import ROOT, Settings
from main import create_app
from ml.dataset import load_dataset
from schemas.churn import CUSTOMER_EXAMPLE


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(ROOT / "data/churn_dataset.csv")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        dataset_path=ROOT / "data/churn_dataset.csv",
        model_path=tmp_path / "models/churn_model.joblib",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def customer():
    return dict(CUSTOMER_EXAMPLE)
