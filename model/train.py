import os
import sys

# Repo root on path so `python model/train.py` finds `config.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import boto3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import datetime
from config import (
    WEEKLY_FREQ,
    DB_CONNECT_TIMEOUT_SECONDS,
    AWS_CONNECT_TIMEOUT_SECONDS,
    AWS_READ_TIMEOUT_SECONDS,
    AWS_MAX_RETRIES,
)
from botocore.config import Config

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS})


def load_transactions(user_id):
    """Load transactions from PostgreSQL into a DataFrame"""
    query = """
        SELECT date, amount, category
        FROM transactions
        WHERE user_id = %(user_id)s
        ORDER BY date
    """
    df = pd.read_sql(query, engine, params={"user_id": user_id})
    df["date"] = pd.to_datetime(df["date"])
    return df


def prepare_weekly(df, category):
    """Aggregate to weekly spend for a single category"""
    cat_df = df[df["category"] == category].copy()

    cat_df["week"] = cat_df["date"].dt.to_period("W").dt.to_timestamp()
    weekly = (
        cat_df.groupby("week")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"week": "ds", "amount": "y"})
    )

    # Fill missing weeks with zero
    full_range = pd.date_range(
        start=weekly["ds"].min(),
        end=weekly["ds"].max(),
        freq=WEEKLY_FREQ
    )
    weekly = (
        weekly.set_index("ds")
        .reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "ds"})
    )

    return weekly


def train_model(weekly_df):
    """Fit a Prophet model on weekly spend data"""
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(weekly_df)
    return model


def save_model_to_s3(user_id, category, model, *, log_save: bool = True):
    """Serialize and store trained model in S3. Returns S3 object key."""
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
    s3.put_object(
        Bucket=os.getenv("S3_BUCKET"),
        Key=key,
        Body=pickle.dumps(model)
    )
    if log_save:
        print(f"Model saved to S3: {key}")
    return key


def train_all_models_for_user(user_id: int, *, quiet: bool = False) -> dict:
    """
    Train and upload one Prophet model per category present in DB for this user.
    Returns a dict suitable for API reporting (trained / skipped / errors).
    """
    trained: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []

    df = load_transactions(user_id)
    if df.empty:
        return {
            "trained": trained,
            "skipped": skipped,
            "errors": errors,
            "transaction_count": 0,
            "message": "No transactions found for this user.",
        }

    categories = df["category"].unique()
    for category in categories:
        try:
            weekly = prepare_weekly(df, category)
            n_weeks = len(weekly)
            if n_weeks < 4:
                skipped.append({
                    "category": category,
                    "reason": "not_enough_weeks",
                    "weeks": n_weeks,
                })
                if not quiet:
                    print(f"Skipping {category} — not enough data ({n_weeks} weeks)")
                continue
            if not quiet:
                print(f"Training model for: {category}")
            model = train_model(weekly)
            key = save_model_to_s3(
                user_id, category, model, log_save=not quiet
            )
            trained.append({
                "category": category,
                "weeks": n_weeks,
                "s3_key": key,
            })
        except Exception as exc:
            errors.append({
                "category": category,
                "message": str(exc),
            })
            if not quiet:
                print(f"Error training {category}: {exc}")

    return {
        "trained": trained,
        "skipped": skipped,
        "errors": errors,
        "transaction_count": len(df),
        "message": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prophet models per category and upload to S3.")
    parser.add_argument("--user-id", type=int, default=2, help="users.id to train on")
    args = parser.parse_args()
    user_id = args.user_id

    print("Loading transactions...")
    out = train_all_models_for_user(user_id, quiet=False)
    print(f"Loaded {out['transaction_count']} transactions")
    print(f"Trained: {[t['category'] for t in out['trained']]}")
    print("All models trained and saved.")