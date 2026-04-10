import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import boto3
import pandas as pd
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config
from datetime import datetime, timedelta
from config import (
    WEEKLY_FREQ,
    BUDGETS,
    DB_CONNECT_TIMEOUT_SECONDS,
    AWS_CONNECT_TIMEOUT_SECONDS,
    AWS_READ_TIMEOUT_SECONDS,
    AWS_MAX_RETRIES,
    ANOMALY_ZSCORE_THRESHOLD,
    ANOMALY_MIN_ABS_DELTA,
    ANOMALY_MIN_PCT_DELTA,
)

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS})
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when a category model cannot be loaded from S3."""

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


def load_model_from_s3(user_id, category):
    """Load a trained Prophet model from S3"""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
        config=Config(
            connect_timeout=AWS_CONNECT_TIMEOUT_SECONDS,
            read_timeout=AWS_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": AWS_MAX_RETRIES, "mode": "standard"},
        ),
    )
    key = f"models/{user_id}/{category.replace(' ', '_')}.pkl"
    try:
        response = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=key)
    except ClientError as exc:
        aws_code = exc.response.get("Error", {}).get("Code")
        if aws_code in {"NoSuchKey", "404"}:
            raise ModelLoadError(
                f"Model not found for '{category}' at key '{key}'",
                "MODEL_NOT_FOUND"
            ) from exc
        raise ModelLoadError(
            f"Could not load model for '{category}' from key '{key}'",
            "MODEL_LOAD_S3_ERROR"
        ) from exc
    except BotoCoreError as exc:
        raise ModelLoadError(
            f"Could not load model for '{category}' from key '{key}'",
            "MODEL_LOAD_S3_ERROR"
        ) from exc
    try:
        return pickle.loads(response["Body"].read())
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ModelLoadError(
            f"Model file is corrupt for '{category}' at key '{key}'",
            "MODEL_DESERIALIZE_ERROR"
        ) from exc


def get_current_week_spend(user_id, category):
    """Get actual spend for the latest complete Monday-anchored week."""
    query = """
        WITH latest AS (
            SELECT MAX(date) AS latest_date
            FROM transactions
            WHERE user_id = %(user_id)s
        ),
        week_bounds AS (
            SELECT
                date_trunc('week', latest_date)::date AS week_start,
                (date_trunc('week', latest_date) + INTERVAL '7 days')::date AS week_end
            FROM latest
        )
        SELECT COALESCE(SUM(t.amount), 0) as total
        FROM transactions t
        CROSS JOIN week_bounds wb
        WHERE t.user_id = %(user_id)s
        AND t.category = %(category)s
        AND t.date >= wb.week_start
        AND t.date < wb.week_end
    """
    result = pd.read_sql(
        query,
        engine,
        params={"user_id": user_id, "category": category}
    )
    return float(result["total"].iloc[0] or 0)


def get_forecast(model):
    """Generate forecast for next week"""
    future = model.make_future_dataframe(periods=1, freq=WEEKLY_FREQ)
    forecast = model.predict(future)
    last = forecast.iloc[-1]
    return {
    "yhat": max(0, round(float(last["yhat"]), 2)),
    "yhat_lower": max(0, round(float(last["yhat_lower"]), 2)),
    "yhat_upper": max(0, round(float(last["yhat_upper"]), 2))
}


def detect_anomaly(actual, forecast):
    """Compare actual spend to forecast interval using calibrated thresholds."""
    yhat = forecast["yhat"]
    upper = forecast["yhat_upper"]
    lower = forecast["yhat_lower"]
    half_width = max((upper - lower) / 2.0, 1e-6)
    zscore = (actual - yhat) / half_width
    abs_delta = abs(actual - yhat)
    pct_delta = (abs_delta / yhat) if yhat > 0 else 0.0
    passes_min_delta = (
        abs_delta >= ANOMALY_MIN_ABS_DELTA
        and pct_delta >= ANOMALY_MIN_PCT_DELTA
    )

    if actual > upper and zscore >= ANOMALY_ZSCORE_THRESHOLD and passes_min_delta:
        severity_pct = round(((actual - yhat) / yhat) * 100, 1) if yhat > 0 else 0
        return {
            "flag": True,
            "direction": "over",
            "actual": actual,
            "expected": yhat,
            "upper": upper,
            "severity_pct": severity_pct,
            "anomaly_score": round(zscore, 2),
        }
    elif actual < lower and actual > 0 and zscore <= -ANOMALY_ZSCORE_THRESHOLD and passes_min_delta:
        return {
            "flag": True,
            "direction": "under",
            "actual": actual,
            "expected": yhat,
            "lower": lower,
            "severity_pct": None,
            "anomaly_score": round(zscore, 2),
        }
    return {
        "flag": False,
        "actual": actual,
        "expected": yhat,
        "anomaly_score": round(zscore, 2),
    }


def generate_alert_message(anomaly, category, monthly_budget=None):
    """Generate a human readable decision-support alert"""
    if not anomaly["flag"]:
        return None

    if anomaly["direction"] == "over":
        msg = (
            f"Your {category} spend this week (${anomaly['actual']}) "
            f"is {anomaly['severity_pct']}% above your expected baseline "
            f"(${anomaly['expected']})."
        )
        if monthly_budget:
            projected = anomaly["actual"] * 4
            gap = round(projected - monthly_budget, 2)
            if gap > 0:
                msg += f" At this pace, you'll exceed your monthly budget by ${gap}."
        return msg

    if anomaly["direction"] == "under":
        return (
            f"Your {category} spend this week (${anomaly['actual']}) "
            f"is unusually low compared to your baseline (${anomaly['expected']}). "
            f"You're tracking well under budget."
        )


def run_anomaly_check(user_id, categories, budgets=None, request_id=None):
    """Run full anomaly check across all categories"""
    budgets = budgets or {}
    results = []
    errors = []

    for category in categories:
        try:
            model = load_model_from_s3(user_id, category)
            forecast = get_forecast(model)
            actual = get_current_week_spend(user_id, category)
            anomaly = detect_anomaly(actual, forecast)
            alert = generate_alert_message(
                anomaly,
                category,
                monthly_budget=budgets.get(category)
            )

            results.append({
                "category": category,
                "actual": round(actual, 2),
                "expected": round(forecast["yhat"], 2),
                "upper_bound": forecast["yhat_upper"],
                "flag": anomaly["flag"],
                "direction": anomaly.get("direction"),
                "severity_pct": anomaly.get("severity_pct"),
                "anomaly_score": anomaly.get("anomaly_score"),
                "alert": alert
            })

            if alert:
                logger.warning(
                    "anomaly_alert",
                    extra={
                        "request_id": request_id,
                        "user_id": user_id,
                        "category": category,
                        "alert": alert,
                    },
                )
            else:
                logger.info(
                    "anomaly_ok",
                    extra={
                        "request_id": request_id,
                        "user_id": user_id,
                        "category": category,
                        "actual": round(actual, 2),
                    },
                )

        except ModelLoadError as e:
            error_msg = str(e)
            logger.exception(
                "anomaly_model_load_failed",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "category": category,
                    "error_code": e.error_code,
                },
            )
            errors.append({
                "category": category,
                "error_type": "model_load_error",
                "error_code": e.error_code,
                "message": error_msg
            })
        except Exception as e:
            error_msg = f"Unexpected error for '{category}': {e}"
            logger.exception(
                "anomaly_processing_failed",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "category": category,
                },
            )
            errors.append({
                "category": category,
                "error_type": "processing_error",
                "error_code": "CATEGORY_PROCESSING_ERROR",
                "message": error_msg
            })

    return {"results": results, "errors": errors}


if __name__ == "__main__":
    user_id = 2
    categories = [
        "Food and Drink",
        "Travel",
        "Shopping",
        "Groceries",
        "Entertainment",
        "Health",
        "Utilities",
        "Uncategorized"
    ]

    budgets = BUDGETS

    print("Running anomaly detection...\n")
    payload = run_anomaly_check(user_id, categories, budgets)

    print("\n--- Summary ---")
    for r in payload["results"]:
        print(f"{r['category']}: actual=${r['actual']} | expected=${r['expected']} | flagged={r['flag']}")