from dataclasses import replace
from pathlib import Path

import pytest

from core.errors import ServiceError
from core.service import ChurnService
from ml.dataset import load_dataset
from ml.storage import load_churn_model, save_churn_model
from schemas.churn import TrainingConfigChurn


def test_atomic_save_failure_keeps_previous_file(settings, monkeypatch):
    state = ChurnService(settings)
    state.startup()
    state.train(TrainingConfigChurn())
    before = settings.model_path.read_bytes()

    def fail_dump(artifact, path):
        path.write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr("ml.storage.joblib.dump", fail_dump)
    with pytest.raises(ServiceError, match="сохранить"):
        save_churn_model(state.artifact, settings.model_path)
    assert settings.model_path.read_bytes() == before
    assert list(settings.model_path.parent.glob("*.tmp")) == []


def test_loading_incompatible_schema_rejected(settings):
    state = ChurnService(settings)
    state.startup()
    state.train(TrainingConfigChurn())
    incompatible = replace(state.artifact, feature_names=["wrong"])
    save_churn_model(incompatible, settings.model_path)
    with pytest.raises(ServiceError, match="загрузить"):
        load_churn_model(settings.model_path)


@pytest.mark.parametrize("loader", [load_dataset, load_churn_model])
def test_unreadable_directory_produces_service_error(monkeypatch, tmp_path, loader):
    def denied(self):
        raise PermissionError("Access denied")

    monkeypatch.setattr(Path, "is_file", denied)
    with pytest.raises(ServiceError):
        loader(tmp_path / "unreadable")
