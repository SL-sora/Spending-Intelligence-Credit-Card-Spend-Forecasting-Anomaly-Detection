import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from prophet import Prophet
from config import (
    CATEGORIES,
    SUMMARY_CSV_DIR,
    VALIDATION_IMG_DIR,
    WEEKLY_FREQ,
    DB_CONNECT_TIMEOUT_SECONDS,
    TARGET_INTERVAL_COVERAGE,
)

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS})


def load_weekly(user_id, category):
    """Load and aggregate weekly spend for a category"""
    query = """
        SELECT date, amount
        FROM transactions
        WHERE user_id = %(user_id)s
        AND category = %(category)s
        ORDER BY date
    """
    df = pd.read_sql(
        query,
        engine,
        params={"user_id": user_id, "category": category}
    )
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").dt.to_timestamp()

    weekly = (
        df.groupby("week")["amount"]
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


def walk_forward_validate(weekly_df, min_train_weeks=12):
    """
    Walk-forward validation:
    Train on weeks 1 to N, predict week N+1.
    Repeat for all available weeks.
    """
    weeks = sorted(weekly_df["ds"].unique())
    errors = []
    results = []

    for i in range(min_train_weeks, len(weeks)):
        train = weekly_df[weekly_df["ds"] < weeks[i]].copy()
        actual_row = weekly_df[weekly_df["ds"] == weeks[i]]

        if actual_row.empty:
            continue

        actual = float(actual_row["y"].values[0])

        # Train model on historical data up to this point
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=TARGET_INTERVAL_COVERAGE
        )
        model.fit(train)

        # Predict one week ahead
        future = model.make_future_dataframe(periods=1, freq=WEEKLY_FREQ)
        forecast = model.predict(future)
        predicted = max(0, float(forecast.iloc[-1]["yhat"]))
        upper = max(0, float(forecast.iloc[-1]["yhat_upper"]))
        lower = max(0, float(forecast.iloc[-1]["yhat_lower"]))

        error = abs(actual - predicted)
        pct_error = (error / actual * 100) if actual > 0 else 0
        in_interval = lower <= actual <= upper

        errors.append(error)
        results.append({
            "week": weeks[i],
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "upper": round(upper, 2),
            "lower": round(lower, 2),
            "error": round(error, 2),
            "pct_error": round(pct_error, 1),
            "in_interval": in_interval
        })

    return pd.DataFrame(results), errors


def print_summary(category, results_df, errors):
    """Print validation metrics"""
    metrics = build_metrics(category, results_df, errors)
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    coverage = metrics["coverage"]
    median_pct_error = metrics["median_pct_error"]

    print(f"\n{'='*45}")
    print(f"Category: {category}")
    print(f"{'='*45}")
    print(f"Weeks validated:        {len(results_df)}")
    print(f"MAE (Mean Abs Error):   ${mae}")
    print(f"RMSE:                   ${rmse}")
    print(f"Median % Error:         {median_pct_error}%")
    print(f"Interval Coverage:      {coverage}%")
    target_pct = int(TARGET_INTERVAL_COVERAGE * 100)
    print(f"  (% of actual weeks that fell within")
    print(f"   the {target_pct}% confidence interval)")

    return metrics


def build_metrics(category, results_df, errors):
    """Build comparable validation metrics for one category."""
    mae = round(np.mean(errors), 2)
    rmse = round(np.sqrt(np.mean(np.array(errors) ** 2)), 2)
    coverage = round(results_df["in_interval"].mean() * 100, 1)
    median_pct_error = round(results_df["pct_error"].median(), 1)
    return {
        "category": category,
        "mae": mae,
        "rmse": rmse,
        "median_pct_error": median_pct_error,
        "coverage": coverage,
        "weeks_validated": len(results_df)
    }


def plot_validation(category, results_df):
    """Plot actual vs predicted with confidence interval"""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(
        results_df["week"],
        results_df["lower"],
        results_df["upper"],
        alpha=0.2,
        color="blue",
        label="90% confidence interval"
    )
    ax.plot(
        results_df["week"],
        results_df["actual"],
        color="black",
        linewidth=1.5,
        label="Actual spend",
        marker="o",
        markersize=3
    )
    ax.plot(
        results_df["week"],
        results_df["predicted"],
        color="blue",
        linewidth=1.5,
        linestyle="--",
        label="Predicted spend"
    )

    # Flag weeks where actual fell outside interval
    outside = results_df[~results_df["in_interval"]]
    ax.scatter(
        outside["week"],
        outside["actual"],
        color="red",
        zorder=5,
        s=50,
        label="Outside interval (anomaly candidate)"
    )

    ax.set_title(
        f"Walk-Forward Validation — {category}",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly Spend ($)")
    ax.legend()
    plt.tight_layout()

    os.makedirs(VALIDATION_IMG_DIR, exist_ok=True)
    filename = f"validation_{category.replace(' ', '_')}.png"
    out_path = os.path.join(VALIDATION_IMG_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    user_id = int(os.getenv("VALIDATION_USER_ID", "1"))
    categories = list(CATEGORIES)

    all_metrics = []

    for category in categories:
        try:
            weekly = load_weekly(user_id, category)

            if len(weekly) < 16:
                print(f"Skipping {category} — not enough weeks ({len(weekly)})")
                continue

            print(f"\nValidating: {category} ({len(weekly)} weeks of data)...")
            results_df, errors = walk_forward_validate(weekly)
            metrics = print_summary(category, results_df, errors)
            all_metrics.append(metrics)
            plot_validation(category, results_df)

        except Exception as e:
            print(f"Could not validate {category}: {e}")

    # Final summary table
    print(f"\n{'='*65}")
    print("VALIDATION SUMMARY TABLE")
    print(f"{'='*65}")
    summary_df = pd.DataFrame(all_metrics)
    print(summary_df.to_string(index=False))

    os.makedirs(SUMMARY_CSV_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_CSV_DIR, "validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")