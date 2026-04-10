import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EDA_IMG_DIR

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# Load all transactions for user 2
df = pd.read_sql("""
    SELECT date, amount, category, merchant_name
    FROM transactions
    WHERE user_id = 2
    ORDER BY date
""", engine)

df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.to_period("W").dt.to_timestamp()

print(f"Total transactions: {len(df)}")
print(f"\nCategory breakdown:")
print(df["category"].value_counts())
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSpend summary by category:")
print(df.groupby("category")["amount"].describe().round(2))

# --- Plot 1: Total spend by category ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Spending Intelligence — EDA", fontsize=16, fontweight="bold")

category_spend = df.groupby("category")["amount"].sum().sort_values(ascending=False)
axes[0, 0].bar(category_spend.index, category_spend.values, color="steelblue")
axes[0, 0].set_title("Total Spend by Category")
axes[0, 0].set_xlabel("Category")
axes[0, 0].set_ylabel("Total Spend ($)")
axes[0, 0].tick_params(axis="x", rotation=45)

# --- Plot 2: Transaction count by category ---
category_count = df["category"].value_counts()
axes[0, 1].bar(category_count.index, category_count.values, color="coral")
axes[0, 1].set_title("Transaction Count by Category")
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("Number of Transactions")
axes[0, 1].tick_params(axis="x", rotation=45)

# --- Plot 3: Weekly spend over time (top 3 categories) ---
top_categories = category_spend.head(3).index.tolist()
weekly_spend = df[df["category"].isin(top_categories)].groupby(
    ["week", "category"]
)["amount"].sum().reset_index()

for cat in top_categories:
    cat_data = weekly_spend[weekly_spend["category"] == cat]
    axes[1, 0].plot(cat_data["week"], cat_data["amount"], label=cat, marker="o", markersize=3)

axes[1, 0].set_title("Weekly Spend Over Time (Top 3 Categories)")
axes[1, 0].set_xlabel("Week")
axes[1, 0].set_ylabel("Weekly Spend ($)")
axes[1, 0].legend()
axes[1, 0].tick_params(axis="x", rotation=45)

# --- Plot 4: Spend distribution boxplot ---
top_cats_df = df[df["category"].isin(top_categories)]
top_cats_df.boxplot(column="amount", by="category", ax=axes[1, 1])
axes[1, 1].set_title("Spend Distribution by Category")
axes[1, 1].set_xlabel("Category")
axes[1, 1].set_ylabel("Transaction Amount ($)")
plt.suptitle("")

plt.tight_layout()
os.makedirs(EDA_IMG_DIR, exist_ok=True)
out_path = os.path.join(EDA_IMG_DIR, "eda_output.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nEDA chart saved as {out_path}")