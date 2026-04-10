import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EDA_IMG_DIR

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

os.makedirs(EDA_IMG_DIR, exist_ok=True)

df = pd.read_sql("""
    SELECT date, amount, category, merchant_name
    FROM transactions
    WHERE user_id = 2
    ORDER BY date
""", engine)

df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.to_period("W").dt.to_timestamp()
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

sns.set_theme(style="whitegrid")
CATEGORIES = df["category"].unique().tolist()
COLORS = sns.color_palette("tab10", len(CATEGORIES))
CAT_COLOR = dict(zip(CATEGORIES, COLORS))

# ============================================================
# PLOT 1: Spend Distribution Per Category (violin + strip)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Cap at 95th percentile so outliers don't squash the chart
p95 = df["amount"].quantile(0.95)
df_capped = df[df["amount"] <= p95]

sns.violinplot(
    data=df_capped,
    x="category",
    y="amount",
    palette="tab10",
    inner="quartile",
    ax=ax
)
ax.set_title(
    "Spend Distribution Per Category (capped at 95th percentile)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Category", fontsize=11)
ax.set_ylabel("Transaction Amount ($)", fontsize=11)
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
p1 = os.path.join(EDA_IMG_DIR, "eda_1_distribution.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p1}")


# ============================================================
# PLOT 2: Weekly Spend Trends Over Time Per Category
# ============================================================
weekly = (
    df.groupby(["week", "category"])["amount"]
    .sum()
    .reset_index()
)

categories_sorted = (
    df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)

fig, axes = plt.subplots(
    nrows=4, ncols=2,
    figsize=(16, 20),
    sharex=True
)
axes = axes.flatten()

for i, cat in enumerate(categories_sorted):
    cat_data = weekly[weekly["category"] == cat].copy()
    cat_data = cat_data.sort_values("week")

    # Rolling 4-week average
    cat_data["rolling_avg"] = cat_data["amount"].rolling(4, min_periods=1).mean()

    ax = axes[i]
    ax.fill_between(
        cat_data["week"],
        cat_data["amount"],
        alpha=0.3,
        color=CAT_COLOR[cat]
    )
    ax.plot(
        cat_data["week"],
        cat_data["amount"],
        color=CAT_COLOR[cat],
        linewidth=1,
        label="Weekly spend"
    )
    ax.plot(
        cat_data["week"],
        cat_data["rolling_avg"],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="4-week avg"
    )
    ax.set_title(cat, fontsize=12, fontweight="bold")
    ax.set_ylabel("Spend ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

# Hide empty subplot if odd number of categories
if len(categories_sorted) < len(axes):
    axes[-1].set_visible(False)

fig.suptitle(
    "Weekly Spend Trends by Category (with 4-week rolling average)",
    fontsize=15, fontweight="bold", y=1.01
)
plt.tight_layout()
p2 = os.path.join(EDA_IMG_DIR, "eda_2_weekly_trends.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")


# ============================================================
# PLOT 3: Transaction Count vs Total Spend Per Category
# ============================================================
summary = df.groupby("category").agg(
    total_spend=("amount", "sum"),
    transaction_count=("amount", "count"),
    avg_transaction=("amount", "mean")
).reset_index().sort_values("total_spend", ascending=False)

print("\n--- Transaction Count vs Spend ---")
print(summary.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Transaction Count vs Spend Analysis",
    fontsize=14, fontweight="bold"
)

# Total spend bar
colors = [CAT_COLOR[c] for c in summary["category"]]
axes[0].barh(summary["category"], summary["total_spend"], color=colors)
axes[0].set_title("Total Spend by Category")
axes[0].set_xlabel("Total Spend ($)")

# Transaction count bar
axes[1].barh(summary["category"], summary["transaction_count"], color=colors)
axes[1].set_title("Transaction Count by Category")
axes[1].set_xlabel("Number of Transactions")

# Average transaction size
axes[2].barh(summary["category"], summary["avg_transaction"], color=colors)
axes[2].set_title("Avg Transaction Size by Category")
axes[2].set_xlabel("Average Amount ($)")

plt.tight_layout()
p3 = os.path.join(EDA_IMG_DIR, "eda_3_count_vs_spend.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p3}")

print("\nAll EDA charts saved.")