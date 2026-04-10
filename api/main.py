from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import logging
from uuid import uuid4
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.anomaly import run_anomaly_check
from model.train import load_transactions, train_all_models_for_user
from model.validate import load_weekly, walk_forward_validate, build_metrics
from config import CATEGORIES, BUDGETS
from ingest.csv_statement import import_statement_csv
from api.schemas import (
    RootResponse,
    AnomalyCheckResponse,
    ModelQualityResponse,
    CsvIngestResponse,
    CsvAnomalyReportResponse,
)

app = FastAPI(title="Spending Intelligence API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_model=RootResponse)
def root():
    return {"status": "Spending Intelligence API is running"}


def get_request_id(request: Request):
    return request.headers.get("x-request-id") or str(uuid4())


@app.get("/anomaly/check/{user_id}", response_model=AnomalyCheckResponse)
def check_anomalies(user_id: int, request: Request):
    request_id = get_request_id(request)
    payload = run_anomaly_check(user_id, CATEGORIES, BUDGETS, request_id=request_id)
    if not payload["results"] and payload["errors"]:
        raise HTTPException(
            status_code=503,
            detail={
                "request_id": request_id,
                "message": "No category checks completed successfully.",
                "errors": payload["errors"]
            }
        )
    return {"request_id": request_id, "user_id": user_id, **payload}

@app.get("/anomaly/check/{user_id}/{category}", response_model=AnomalyCheckResponse)
def check_single_category(user_id: int, category: str, request: Request):
    request_id = get_request_id(request)
    payload = run_anomaly_check(user_id, [category], BUDGETS, request_id=request_id)
    if not payload["results"] and payload["errors"]:
        error = payload["errors"][0]
        detail = {"request_id": request_id, **error}
        if error["error_code"] == "MODEL_NOT_FOUND":
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=503, detail=detail)
    return {"request_id": request_id, "user_id": user_id, **payload}


@app.get("/model/quality/{user_id}", response_model=ModelQualityResponse)
def model_quality_report(user_id: int, request: Request):
    request_id = get_request_id(request)
    metrics = []
    errors = []

    for category in CATEGORIES:
        try:
            weekly = load_weekly(user_id, category)
            if len(weekly) < 16:
                errors.append({
                    "category": category,
                    "error_type": "not_enough_data",
                    "message": f"Need at least 16 weeks, got {len(weekly)}."
                })
                continue
            results_df, err_values = walk_forward_validate(weekly)
            metrics.append(build_metrics(category, results_df, err_values))
        except Exception as exc:
            errors.append({
                "category": category,
                "error_type": "validation_error",
                "message": str(exc)
            })

    if not metrics and errors:
        raise HTTPException(
            status_code=503,
            detail={
                "request_id": request_id,
                "message": "Could not produce quality metrics.",
                "errors": errors,
            },
        )

    return {
        "request_id": request_id,
        "user_id": user_id,
        "metrics": metrics,
        "errors": errors,
    }


@app.post("/ingest/csv", response_model=CsvIngestResponse)
async def ingest_csv(
    request: Request,
    user_id: int = Form(..., description="Target users.id"),
    file: UploadFile = File(..., description="Statement CSV"),
    debits_negative: bool = Form(
        True,
        description="If true, only negative amounts are treated as spending (typical bank export).",
    ),
    account_id: str = Form("csv_import"),
):
    request_id = get_request_id(request)
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"request_id": request_id, "message": "Empty file."},
        )
    try:
        result = import_statement_csv(
            raw,
            user_id,
            debits_negative=debits_negative,
            account_id=account_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"request_id": request_id, "message": str(exc)},
        ) from exc
    return {
        "request_id": request_id,
        "user_id": user_id,
        "inserted": result["inserted"],
        "message": result.get("message"),
    }


@app.post("/report/csv", response_model=CsvAnomalyReportResponse)
async def csv_anomaly_report(
    request: Request,
    user_id: int = Form(..., description="Target users.id"),
    file: UploadFile = File(..., description="Statement CSV"),
    debits_negative: bool = Form(
        True,
        description="If true, negative amounts are spending when mixed signs exist; all-positive CSVs use positive amounts.",
    ),
    account_id: str = Form("csv_import"),
    retrain_models: bool = Form(
        True,
        description="Retrain Prophet models from DB and upload to S3 before anomaly check.",
    ),
):
    """
    Upload a CSV, insert transactions, optionally retrain models, then run anomaly detection.
    Returns ingest stats, training stats, per-category anomaly results, and a short summary.
    """
    request_id = get_request_id(request)
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"request_id": request_id, "message": "Empty file."},
        )

    try:
        ingest_result = import_statement_csv(
            raw,
            user_id,
            debits_negative=debits_negative,
            account_id=account_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"request_id": request_id, "message": str(exc)},
        ) from exc

    if retrain_models:
        train_out = train_all_models_for_user(user_id, quiet=True)
    else:
        try:
            tx_df = load_transactions(user_id)
            tx_n = len(tx_df)
        except Exception:
            tx_n = 0
        train_out = {
            "trained": [],
            "skipped": [],
            "errors": [],
            "transaction_count": tx_n,
            "message": "Retrain skipped (retrain_models=false).",
        }

    payload = run_anomaly_check(user_id, CATEGORIES, BUDGETS, request_id=request_id)
    results = payload["results"]
    err_list = payload["errors"]

    flagged_count = sum(1 for r in results if r.get("flag"))
    ok_count = sum(1 for r in results if not r.get("flag"))
    alerts = [r["alert"] for r in results if r.get("alert")]

    return {
        "request_id": request_id,
        "user_id": user_id,
        "ingest": {
            "inserted": ingest_result["inserted"],
            "message": ingest_result.get("message"),
        },
        "training": {
            "transaction_count": train_out["transaction_count"],
            "trained": train_out["trained"],
            "skipped": train_out["skipped"],
            "errors": train_out["errors"],
            "message": train_out.get("message"),
        },
        "summary": {
            "categories_evaluated": len(results),
            "categories_failed": len(err_list),
            "flagged_count": flagged_count,
            "ok_count": ok_count,
            "alerts": alerts,
        },
        "results": results,
        "errors": err_list,
    }