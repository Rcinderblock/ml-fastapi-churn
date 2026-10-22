"""Состояние одного процесса: датасет, текущая модель и история обучений."""

import logging
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

import pandas as pd

from core.config import RANDOM_STATE, TEST_SIZE, Settings
from core.errors import ServiceError
from ml.dataset import (
    dataset_fingerprint,
    load_dataset,
    split_dataset,
)
from ml.pipeline import evaluate_churn_model, train_churn_model
from ml.storage import ModelArtifact, load_churn_model, save_churn_model
from schemas.churn import (
    FEATURE_NAMES,
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainingRecordChurn,
)

logger = logging.getLogger(__name__)


class ChurnService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dataset: pd.DataFrame | None = None
        self.artifact: ModelArtifact | None = None
        self.dataset_error: str | None = None
        self.model_error: str | None = None
        self._train_lock = Lock()

    def startup(self):
        try:
            self.dataset = load_dataset(self.settings.dataset_path)
        except ServiceError as exc:
            self.dataset_error = exc.code
            logger.warning("Датасет недоступен: %s", exc.code)
        try:
            self.artifact = load_churn_model(self.settings.model_path)
            if self.artifact is not None:
                logger.info("Модель загружена: %s", self.artifact.record.training_id)
        except ServiceError as exc:
            self.model_error = exc.code
            logger.warning("Сохранённая модель недоступна: %s", exc.code)

    def require_dataset(self) -> pd.DataFrame:
        if self.dataset is None:
            raise ServiceError(
                "DATASET_NOT_LOADED",
                "Датасет не загружен: проверьте CSV и перезапустите сервис",
                400,
                {"reason": self.dataset_error},
            )
        if self.dataset.empty:
            raise ServiceError("DATASET_EMPTY", "Датасет пуст")
        return self.dataset

    def train(self, config: TrainingConfigChurn) -> TrainingRecordChurn:
        if not self._train_lock.acquire(blocking=False):
            raise ServiceError("TRAINING_IN_PROGRESS", "Обучение уже выполняется", 409)
        try:
            frame = self.require_dataset()
            split = split_dataset(frame)
            logger.info(
                "Начало обучения: model=%s rows=%d", config.model_type, len(frame)
            )
            try:
                pipeline = train_churn_model(split.train, config)
                metrics = evaluate_churn_model(pipeline, split.test)
            except (ValueError, TypeError) as exc:
                raise ServiceError(
                    "TRAINING_FAILED", "Не удалось обучить модель на этих данных", 422
                ) from exc
            record = TrainingRecordChurn(
                training_id=str(uuid4()),
                timestamp=datetime.now(UTC).isoformat(),
                model_type=config.model_type,
                hyperparameters=dict(config.hyperparameters),
                metrics=metrics,
                dataset_sha256=dataset_fingerprint(frame),
                train_rows=len(split.train),
                test_rows=len(split.test),
                random_state=RANDOM_STATE,
                test_size=TEST_SIZE,
            )
            previous = self.artifact.history if self.artifact else []
            artifact = ModelArtifact(
                pipeline=pipeline,
                record=record,
                history=[*previous, record],
                feature_names=list(FEATURE_NAMES),
            )
            save_churn_model(artifact, self.settings.model_path)
            # Старая модель обслуживает запросы, пока новая обучается и сохраняется.
            self.artifact = artifact
            self.model_error = None
            logger.info(
                "Обучение завершено: model=%s accuracy=%.4f f1=%.4f",
                record.model_type,
                metrics.accuracy,
                metrics.f1,
            )
            return record
        finally:
            self._train_lock.release()

    def predict(
        self, objects: list[FeatureVectorChurn]
    ) -> list[PredictionResponseChurn]:
        artifact = self.artifact
        if artifact is None:
            raise ServiceError(
                "MODEL_NOT_TRAINED",
                "Сначала обучите модель через POST /model/train",
                503,
            )
        features = pd.DataFrame(
            [obj.model_dump() for obj in objects], columns=FEATURE_NAMES
        )
        try:
            classes = [int(value) for value in artifact.pipeline.classes_]
            labels = artifact.pipeline.predict(features)
            probabilities = artifact.pipeline.predict_proba(features)
            result = [
                PredictionResponseChurn(
                    churn=int(label),
                    probabilities={
                        str(cls): float(probability)
                        for cls, probability in zip(classes, row, strict=True)
                    },
                )
                for label, row in zip(labels, probabilities, strict=True)
            ]
        except (ValueError, TypeError, OverflowError) as exc:
            raise ServiceError(
                "PREDICTION_FAILED", "Не удалось обработать значения признаков", 422
            ) from exc
        logger.info(
            "Предсказание: count=%d training_id=%s",
            len(objects),
            artifact.record.training_id,
        )
        return result

    def status(self):
        artifact = self.artifact
        record = artifact.record if artifact else None
        return {
            "trained": artifact is not None,
            "trained_at": record.timestamp if record else None,
            "training_id": record.training_id if record else None,
            "model_type": record.model_type if record else None,
            "hyperparameters": record.hyperparameters if record else None,
            "metrics": record.metrics.model_dump() if record else None,
            "training_in_progress": self._train_lock.locked(),
            "load_error": self.model_error,
        }
