"""Входные признаки, настройки обучения и ответы сервиса."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Region = Literal["europe", "asia", "america", "africa"]
Device = Literal["mobile", "desktop", "tablet"]
Payment = Literal["card", "paypal", "crypto"]
ModelType = Literal["logreg", "random_forest"]
Count = Annotated[int, Field(ge=0)]
Binary = Annotated[int, Field(ge=0, le=1)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]

FEATURE_NAMES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]
NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
TARGET = "churn"

CUSTOMER_EXAMPLE = {
    "monthly_fee": 19.99,
    "usage_hours": 8.5,
    "support_requests": 3,
    "account_age_months": 6,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0,
}


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"examples": [CUSTOMER_EXAMPLE]},
    )

    monthly_fee: NonNegative
    usage_hours: NonNegative
    support_requests: Count
    account_age_months: Count
    failed_payments: Count
    region: Region
    device_type: Device
    payment_method: Payment
    autopay_enabled: Binary


class DatasetRowChurn(FeatureVectorChurn):
    # CSV содержит строки; пропуски признаков заполняет обучаемый Pipeline.
    model_config = ConfigDict(extra="forbid", strict=False, allow_inf_nan=False)

    monthly_fee: NonNegative | None
    usage_hours: NonNegative | None
    support_requests: Count | None
    account_age_months: Count | None
    failed_payments: Count | None
    region: Region | None
    device_type: Device | None
    payment_method: Payment | None
    autopay_enabled: Binary | None
    churn: Binary


class LogregParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    C: Annotated[float, Field(gt=0)] = 1.0
    max_iter: Annotated[int, Field(ge=1, le=10000)] = 1000
    solver: Literal["lbfgs", "liblinear"] = "lbfgs"
    class_weight: Literal["balanced"] | None = "balanced"
    random_state: Annotated[int, Field(ge=0, le=4294967295)] = 42


class ForestParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    n_estimators: Annotated[int, Field(ge=1, le=1000)] = 100
    max_depth: Annotated[int, Field(ge=1, le=100)] | None = None
    min_samples_split: Annotated[int, Field(ge=2, le=1000)] = 2
    min_samples_leaf: Annotated[int, Field(ge=1, le=1000)] = 1
    max_features: Literal["sqrt", "log2"] | None = "sqrt"
    class_weight: Literal["balanced", "balanced_subsample"] | None = "balanced"
    random_state: Annotated[int, Field(ge=0, le=4294967295)] = 42


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_type: ModelType = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_hyperparameters(self):
        schema = LogregParameters if self.model_type == "logreg" else ForestParameters
        self.hyperparameters = schema.model_validate(self.hyperparameters).model_dump()
        return self


class PredictionResponseChurn(BaseModel):
    churn: Literal[0, 1]
    probabilities: dict[Literal["0", "1"], Annotated[float, Field(ge=0, le=1)]]


class MetricsChurn(BaseModel):
    accuracy: Annotated[float, Field(ge=0, le=1)]
    f1: Annotated[float, Field(ge=0, le=1)]
    roc_auc: Annotated[float, Field(ge=0, le=1)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    confusion_matrix: list[list[int]]


class TrainingRecordChurn(BaseModel):
    training_id: str
    timestamp: str
    model_type: ModelType
    hyperparameters: dict[str, Any]
    metrics: MetricsChurn
    dataset_sha256: str
    train_rows: int
    test_rows: int
    random_state: int
    test_size: float


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any = None
