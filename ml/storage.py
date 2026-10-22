"""Атомарное сохранение Pipeline, метаданных и истории одним объектом."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from core.errors import ServiceError
from schemas.churn import FEATURE_NAMES, TrainingRecordChurn


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    record: TrainingRecordChurn
    history: list[TrainingRecordChurn]
    feature_names: list[str]
    sklearn_version: str = sklearn.__version__
    schema_version: int = 1


def save_churn_model(artifact: ModelArtifact, path: Path) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as f:
            temporary = Path(f.name)
        joblib.dump(artifact, temporary)
        # Ошибка записи не заменяет последнюю успешно сохранённую модель.
        os.replace(temporary, path)
    except (OSError, ValueError) as exc:
        raise ServiceError(
            "MODEL_SAVE_FAILED", "Не удалось сохранить модель", 500
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_churn_model(path: Path) -> ModelArtifact | None:
    try:
        if not path.is_file():
            return None
        # Файл создаётся самим сервисом; загрузки произвольных pickle через API нет.
        artifact = joblib.load(path)
        if not isinstance(artifact, ModelArtifact):
            raise ValueError("Неизвестный формат модели")
        if (
            artifact.schema_version != 1
            or artifact.sklearn_version != sklearn.__version__
            or artifact.feature_names != FEATURE_NAMES
            or list(artifact.pipeline.feature_names_in_) != FEATURE_NAMES
            or list(artifact.pipeline.classes_) != [0, 1]
            or not artifact.history
            or artifact.history[-1].training_id != artifact.record.training_id
        ):
            raise ValueError("Модель несовместима с конфигурацией сервиса")
        return artifact
    except Exception as exc:
        raise ServiceError(
            "MODEL_LOAD_FAILED", "Не удалось загрузить сохранённую модель", 500
        ) from exc
