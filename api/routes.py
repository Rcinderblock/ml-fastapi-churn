"""HTTP маршруты просмотра данных, обучения и предсказаний."""

from typing import Annotated

from fastapi import APIRouter, Body, Query, Request

from core.service import ChurnService
from ml.dataset import class_distribution, dataset_fingerprint, split_dataset
from schemas.churn import (
    CUSTOMER_EXAMPLE,
    FEATURE_NAMES,
    TARGET,
    DatasetRowChurn,
    ErrorResponse,
    FeatureVectorChurn,
    ModelType,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainingRecordChurn,
)

router = APIRouter()


def service(request: Request) -> ChurnService:
    return request.app.state.service


@router.get("/", tags=["service"])
def root():
    return {"message": "ml churn service is running"}


@router.get("/health", tags=["service"])
def health(request: Request):
    state = service(request)
    loaded = state.dataset is not None and not state.dataset.empty
    available = state.artifact is not None
    return {
        "status": "ok" if loaded and available else "degraded",
        "model_available": available,
        "dataset_loaded": loaded,
    }


@router.get("/dataset/preview", response_model=list[DatasetRowChurn], tags=["dataset"])
def dataset_preview(request: Request, n: Annotated[int, Query(ge=1, le=100)] = 5):
    frame = service(request).require_dataset().head(n)
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


@router.get("/dataset/info", tags=["dataset"])
def dataset_info(request: Request):
    frame = service(request).require_dataset()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": list(frame.columns),
        "feature_names": FEATURE_NAMES,
        "target": TARGET,
        "class_distribution": class_distribution(frame[TARGET]),
        "missing_values": {
            name: int(value) for name, value in frame.isna().sum().items()
        },
        "dataset_sha256": dataset_fingerprint(frame),
    }


@router.get("/dataset/split-info", tags=["dataset"])
def dataset_split_info(request: Request):
    return split_dataset(service(request).require_dataset()).info()


@router.post(
    "/model/train",
    response_model=TrainingRecordChurn,
    tags=["model"],
    summary="Обучить и сохранить модель",
    description=(
        "Поддерживаются logreg и random_forest. Пропуски и масштабирование "
        "обучаются только на train; метрики считаются на test (20%, seed=42). "
        "Новая модель и история сохраняются автоматически."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Датасет не загружен или пуст"},
        409: {"model": ErrorResponse, "description": "Обучение уже выполняется"},
        422: {"model": ErrorResponse, "description": "Неверные параметры или данные"},
    },
)
def model_train(
    request: Request,
    config: Annotated[
        TrainingConfigChurn | None,
        Body(
            openapi_examples={
                "logreg": {
                    "summary": "Логистическая регрессия",
                    "value": {"model_type": "logreg", "hyperparameters": {"C": 1.0}},
                },
                "forest": {
                    "summary": "Случайный лес",
                    "value": {
                        "model_type": "random_forest",
                        "hyperparameters": {"n_estimators": 100, "max_depth": 8},
                    },
                },
            }
        ),
    ] = None,
):
    return service(request).train(config or TrainingConfigChurn())


@router.get("/model/status", tags=["model"])
def model_status(request: Request):
    return service(request).status()


@router.get("/model/schema", tags=["model"])
def model_schema():
    schema = FeatureVectorChurn.model_json_schema()
    return {
        "feature_order": FEATURE_NAMES,
        "features": [
            {"name": name, **schema["properties"][name]} for name in FEATURE_NAMES
        ],
        "required": schema["required"],
        "additional_properties": False,
    }


@router.get("/model/metrics", tags=["model"])
def model_metrics(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    model_type: ModelType | None = None,
):
    artifact = service(request).artifact
    history = artifact.history if artifact else []
    if model_type:
        history = [row for row in history if row.model_type == model_type]
    return {
        "latest": history[-1].model_dump() if history else None,
        "history": [row.model_dump() for row in reversed(history[-limit:])],
        "total": len(history),
    }


Batch = Annotated[list[FeatureVectorChurn], Body(min_length=1, max_length=1000)]


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    tags=["prediction"],
    summary="Предсказать отток одного клиента или списка клиентов",
    description=(
        "Один объект возвращает один ответ; список из 1–1000 объектов — список ответов "
        "в том же порядке. churn=1 означает уход клиента. probabilities содержит "
        "вероятности классов 0 и 1. Все девять признаков обязательны."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"churn": 1, "probabilities": {"0": 0.25, "1": 0.75}}
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "code": "VALIDATION_ERROR",
                        "message": "Проверьте поля и типы значений запроса",
                        "details": [
                            {"field": ["body", "region"], "type": "literal_error"}
                        ],
                    }
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "code": "MODEL_NOT_TRAINED",
                        "message": "Сначала обучите модель через POST /model/train",
                        "details": None,
                    }
                }
            },
        },
    },
)
def predict(
    request: Request,
    features: Annotated[
        FeatureVectorChurn | Batch,
        Body(
            openapi_examples={
                "single": {"summary": "Один клиент", "value": CUSTOMER_EXAMPLE},
                "batch": {"summary": "Список клиентов", "value": [CUSTOMER_EXAMPLE]},
            }
        ),
    ],
):
    is_batch = isinstance(features, list)
    predictions = service(request).predict(features if is_batch else [features])
    return predictions if is_batch else predictions[0]
