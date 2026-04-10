from typing import Optional
from pydantic import BaseModel


class RootResponse(BaseModel):
    status: str


class AnomalyResult(BaseModel):
    category: str
    actual: float
    expected: float
    upper_bound: float
    flag: bool
    direction: Optional[str] = None
    severity_pct: Optional[float] = None
    anomaly_score: Optional[float] = None
    alert: Optional[str] = None


class AnomalyError(BaseModel):
    category: str
    error_type: str
    error_code: str
    message: str


class AnomalyCheckResponse(BaseModel):
    request_id: str
    user_id: int
    results: list[AnomalyResult]
    errors: list[AnomalyError]


class ModelQualityMetric(BaseModel):
    category: str
    mae: float
    rmse: float
    median_pct_error: float
    coverage: float
    weeks_validated: int


class ModelQualityError(BaseModel):
    category: str
    error_type: str
    message: str


class ModelQualityResponse(BaseModel):
    request_id: str
    user_id: int
    metrics: list[ModelQualityMetric]
    errors: list[ModelQualityError]


class CsvIngestResponse(BaseModel):
    request_id: str
    user_id: int
    inserted: int
    message: str | None = None


class IngestSummary(BaseModel):
    inserted: int
    message: str | None = None


class TrainingTrainedItem(BaseModel):
    category: str
    weeks: int
    s3_key: str | None = None


class TrainingSkippedItem(BaseModel):
    category: str
    reason: str
    weeks: int


class TrainingErrorItem(BaseModel):
    category: str
    message: str


class TrainingSummary(BaseModel):
    transaction_count: int
    trained: list[TrainingTrainedItem]
    skipped: list[TrainingSkippedItem]
    errors: list[TrainingErrorItem]
    message: str | None = None


class AnomalyReportSummary(BaseModel):
    """High-level counts after anomaly check."""
    categories_evaluated: int
    categories_failed: int
    flagged_count: int
    ok_count: int
    alerts: list[str]


class CsvAnomalyReportResponse(BaseModel):
    """
    Full pipeline: CSV ingest → optional retrain → anomaly check + summary.
    """
    request_id: str
    user_id: int
    ingest: IngestSummary
    training: TrainingSummary
    summary: AnomalyReportSummary
    results: list[AnomalyResult]
    errors: list[AnomalyError]
