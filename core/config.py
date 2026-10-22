"""Пути и воспроизводимые параметры разделения данных."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class Settings:
    dataset_path: Path = ROOT / "data" / "churn_dataset.csv"
    model_path: Path = ROOT / "models" / "churn_model.joblib"

    @classmethod
    def from_env(cls):
        return cls(
            dataset_path=Path(
                os.getenv("CHURN_DATASET_PATH", str(ROOT / "data/churn_dataset.csv"))
            ),
            model_path=Path(
                os.getenv("CHURN_MODEL_PATH", str(ROOT / "models/churn_model.joblib"))
            ),
        )
