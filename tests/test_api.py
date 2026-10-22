import json

import pytest
from fastapi.testclient import TestClient

from core.config import Settings
from core.errors import ServiceError
from main import create_app
from schemas.churn import FEATURE_NAMES


def test_root_health_docs_and_dataset(client):
    assert client.get("/").json() == {"message": "ml churn service is running"}
    assert client.get("/health").json() == {
        "status": "degraded",
        "model_available": False,
        "dataset_loaded": True,
    }
    assert client.get("/docs").status_code == 200
    preview = client.get("/dataset/preview?n=3").json()
    assert len(preview) == 3
    assert set(preview[0]) == set(FEATURE_NAMES + ["churn"])
    info = client.get("/dataset/info").json()
    assert (info["rows"], info["columns"]) == (2000, 10)
    assert info["class_distribution"] == {"0": 1597, "1": 403}
    split = client.get("/dataset/split-info").json()
    assert split["train_rows"] == 1600 and split["test_rows"] == 400
    schema = client.get("/model/schema").json()
    assert schema["feature_order"] == FEATURE_NAMES
    assert schema["features"][5]["enum"] == ["europe", "asia", "america", "africa"]
    openapi = client.get("/openapi.json").json()
    predict = openapi["paths"]["/predict"]["post"]
    assert {"200", "422", "503"} <= predict["responses"].keys()
    assert "batch" in predict["requestBody"]["content"]["application/json"]["examples"]


def test_predict_before_training(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_NOT_TRAINED"
    assert set(response.json()) == {"code", "message", "details"}


def test_full_train_predict_history_and_restart(client, settings, customer):
    response = client.post("/model/train", json={"model_type": "logreg"})
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["metrics"]["accuracy"] > 0.5
    assert set(record["metrics"]) == {
        "accuracy",
        "f1",
        "roc_auc",
        "precision",
        "recall",
        "confusion_matrix",
    }
    assert sum(map(sum, record["metrics"]["confusion_matrix"])) == 400
    assert settings.model_path.is_file()
    status = client.get("/model/status").json()
    assert status["trained"] and status["model_type"] == "logreg"
    assert status["trained_at"] == record["timestamp"]
    assert status["hyperparameters"]["C"] == 1.0
    single = client.post("/predict", json=customer)
    assert single.status_code == 200, single.text
    assert single.json()["churn"] in (0, 1)
    assert sum(single.json()["probabilities"].values()) == pytest.approx(1.0)
    reordered = dict(reversed(list(customer.items())))
    batch = client.post("/predict", json=[customer, reordered]).json()
    assert batch == [single.json(), single.json()]
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model/metrics").json()["total"] == 1

    with TestClient(create_app(settings)) as restarted:
        assert (
            restarted.get("/model/status").json()["training_id"]
            == record["training_id"]
        )
        assert restarted.post("/predict", json=customer).json() == single.json()
        assert restarted.get("/model/metrics").json()["total"] == 1
        forest = restarted.post(
            "/model/train",
            json={
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 10},
            },
        )
        assert forest.status_code == 200, forest.text
        assert restarted.get("/model/metrics").json()["total"] == 2
        filtered = restarted.get("/model/metrics?model_type=logreg&limit=1").json()
        assert filtered["total"] == 1
        assert filtered["latest"]["training_id"] == record["training_id"]
        assert restarted.get("/model/status").json()["model_type"] == "random_forest"


@pytest.mark.parametrize(
    "change",
    [
        {"monthly_fee": -1},
        {"monthly_fee": "19.99"},
        {"usage_hours": True},
        {"support_requests": 1.5},
        {"support_requests": True},
        {"failed_payments": -1},
        {"region": "unknown"},
        {"device_type": "watch"},
        {"payment_method": "cash"},
        {"autopay_enabled": 2},
        {"autopay_enabled": True},
        {"churn": 1},
        {"region": None},
    ],
)
def test_invalid_feature_values(client, customer, change):
    response = client.post("/predict", json=customer | change)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert set(response.json()) == {"code", "message", "details"}


@pytest.mark.parametrize("kind", ["missing", "empty_list", "long_list", "wrong_type"])
def test_invalid_feature_shape(client, customer, kind):
    body = customer.copy()
    if kind == "missing":
        body.pop("monthly_fee")
    elif kind == "empty_list":
        body = []
    elif kind == "long_list":
        body = [customer] * 1001
    else:
        body = [1, 2, 3]
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_values_have_json_error(client, customer, invalid):
    body = json.dumps(customer).replace(
        '"usage_hours": 8.5', f'"usage_hours": {invalid}'
    )
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "unknown"},
        {"hyperparameters": {"C": -1}},
        {"hyperparameters": {"max_iter": "100"}},
        {"hyperparameters": {"n_estimators": 20}},
        {"model_type": "random_forest", "hyperparameters": {"n_estimators": 0}},
        {"model_type": "random_forest", "hyperparameters": {"max_depth": -1}},
    ],
)
def test_invalid_training_configuration(client, config):
    response = client.post("/model/train", json=config)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert not client.get("/model/status").json()["trained"]


def test_default_training_body(client):
    response = client.post("/model/train")
    assert response.status_code == 200, response.text
    assert response.json()["model_type"] == "logreg"


@pytest.mark.parametrize("kind", ["missing", "empty", "bad_columns"])
def test_unavailable_dataset(tmp_path, settings, kind):
    path = tmp_path / "dataset.csv"
    if kind == "empty":
        path.write_text("")
    elif kind == "bad_columns":
        path.write_text("bad,column\n1,2\n")
    custom = Settings(dataset_path=path, model_path=settings.model_path)
    with TestClient(create_app(custom)) as client:
        assert not client.get("/health").json()["dataset_loaded"]
        response = client.post("/model/train", json={})
        assert response.status_code == 400
        assert response.json()["code"] == "DATASET_NOT_LOADED"


def test_empty_in_memory_dataset(client):
    client.app.state.service.dataset = client.app.state.service.dataset.head(0)
    response = client.post("/model/train", json={})
    assert response.status_code == 400 and response.json()["code"] == "DATASET_EMPTY"


def test_failed_save_preserves_previous_model(client, customer, monkeypatch):
    first = client.post("/model/train", json={}).json()
    prediction = client.post("/predict", json=customer).json()

    def fail_save(*args):
        raise ServiceError("MODEL_SAVE_FAILED", "Не удалось сохранить модель", 500)

    monkeypatch.setattr("core.service.save_churn_model", fail_save)
    response = client.post("/model/train", json={})
    assert (
        response.status_code == 500 and response.json()["code"] == "MODEL_SAVE_FAILED"
    )
    assert client.get("/model/status").json()["training_id"] == first["training_id"]
    assert client.post("/predict", json=customer).json() == prediction
    assert client.get("/model/metrics").json()["total"] == 1


def test_training_lock_and_http_error(client):
    lock = client.app.state.service._train_lock
    lock.acquire()
    try:
        response = client.post("/model/train", json={})
        assert response.status_code == 409
        assert response.json()["code"] == "TRAINING_IN_PROGRESS"
    finally:
        lock.release()
    assert client.get("/does-not-exist").json()["code"] == "HTTP_ERROR"


def test_corrupt_saved_model_can_be_retrained(settings, customer):
    settings.model_path.parent.mkdir(parents=True)
    settings.model_path.write_bytes(b"invalid joblib")
    with TestClient(create_app(settings)) as client:
        status = client.get("/model/status").json()
        assert not status["trained"] and status["load_error"] == "MODEL_LOAD_FAILED"
        assert client.post("/model/train", json={}).status_code == 200
        assert client.get("/model/status").json()["load_error"] is None
        assert client.post("/predict", json=customer).status_code == 200


def test_unexpected_error_does_not_leak_trace(settings, monkeypatch):
    from core.service import ChurnService

    def crash(self):
        raise RuntimeError("private internal detail")

    monkeypatch.setattr(ChurnService, "status", crash)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.get("/model/status")
        assert response.status_code == 500
        assert response.json() == {
            "code": "INTERNAL_ERROR",
            "message": "Не удалось выполнить операцию",
            "details": None,
        }
        assert "private internal detail" not in response.text
