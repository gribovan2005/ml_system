from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from enum import Enum

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DataSummary(BaseModel):
    rows: int
    columns: int
    features: List[str]
    missing_values: Dict[str, int]

class Metrics(BaseModel):
    MAE: Optional[float] = None
    R2: Optional[float] = None
    Accuracy: Optional[float] = None
    F1: Optional[float] = None

class Plots(BaseModel):
    feature_importance: Optional[str] = None
    predictions_vs_actual: Optional[str] = None
    confusion_matrix: Optional[str] = None

class MLResult(BaseModel):
    task_id: str
    best_model: str
    task_type: str
    metrics: Metrics
    data_summary: DataSummary
    plots: Plots
    model_path: str

class TaskResponse(BaseModel):
    status: TaskStatus
    result: Optional[MLResult] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    task_id: str

class ErrorResponse(BaseModel):
    message: str
    detail: Optional[str] = None
