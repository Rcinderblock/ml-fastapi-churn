"""Проверка CSV, выделение признаков и стратифицированное разделение."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from core.config import RANDOM_STATE, TEST_SIZE
from core.errors import ServiceError, validation_details
from schemas.churn import FEATURE_NAMES, TARGET, DatasetRowChurn

logger = logging.getLogger(__name__)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    expected = FEATURE_NAMES + [TARGET]
    if frame.empty:
        raise ServiceError("DATASET_EMPTY", "Датасет пуст")
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ServiceError(
            "DATASET_INVALID",
            "Столбцы датасета не соответствуют схеме",
            422,
            {"expected": expected, "received": list(frame.columns)},
        )
    records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    validated = []
    for index, row in enumerate(records):
        row = {key: None if value == "" else value for key, value in row.items()}
        try:
            validated.append(DatasetRowChurn.model_validate(row).model_dump())
        except ValidationError as exc:
            raise ServiceError(
                "DATASET_INVALID",
                f"Некорректная строка CSV: {index + 2}",
                422,
                validation_details(exc.errors()),
            ) from exc
    return pd.DataFrame(validated, columns=expected, index=frame.index)


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        if not path.is_file():
            raise ServiceError("DATASET_NOT_FOUND", "CSV файл датасета не найден", 404)
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ServiceError("DATASET_EMPTY", "CSV файл пуст") from exc
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ServiceError(
            "DATASET_INVALID", "Не удалось прочитать CSV файл", 422
        ) from exc
    frame = validate_dataset(frame)
    logger.info("Датасет загружен: rows=%d columns=%d", *frame.shape)
    return frame


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = validate_dataset(frame)
    # Единый порядок столбцов для обучения и предсказания; None -> np.nan.
    features = frame[FEATURE_NAMES].astype(object)
    features = features.where(pd.notna(features), np.nan)
    return features, frame[TARGET].astype(int)


def class_distribution(labels: pd.Series) -> dict[str, int]:
    return {str(label): int((labels == label).sum()) for label in (0, 1)}


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame

    def info(self):
        return {
            "train_rows": len(self.train),
            "test_rows": len(self.test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_class_distribution": class_distribution(self.train[TARGET]),
            "test_class_distribution": class_distribution(self.test[TARGET]),
        }


def split_dataset(frame: pd.DataFrame) -> DatasetSplit:
    _, labels = prepare_data(frame)
    if set(labels.unique()) != {0, 1} or labels.value_counts().min() < 2:
        raise ServiceError(
            "DATASET_INVALID",
            "Для разделения нужны оба класса, минимум по две строки",
            422,
        )
    try:
        train, test = train_test_split(
            frame, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
        )
    except ValueError as exc:
        raise ServiceError(
            "DATASET_INVALID", "Недостаточно данных для разделения", 422
        ) from exc
    return DatasetSplit(train=train, test=test)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
