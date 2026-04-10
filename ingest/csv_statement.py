"""
Import bank/statement CSV rows into PostgreSQL `transactions` for the ML pipeline.

Expected columns (headers are matched case-insensitively):
  - date: transaction_date, clearing_date, date, posting_date, trans_date, ...
  - amount: amount_(usd), amount, amt, debit, outflow, spend
  - category (optional): category, category_name, type (if no Category column), ...
  - merchant (optional): merchant, description, payee, memo, name, ...

Amount sign: if any values are negative, those are treated as spending (typical bank export).
If all amounts are positive (common on corporate cards), positive values are used as spend.
Use --credits-positive / debits_negative=false to force the latter when you also have negative rows.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import BinaryIO, Optional, TextIO, Union

CsvSource = Union[str, Path, BinaryIO, TextIO]

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import CATEGORIES, DB_CONNECT_TIMEOUT_SECONDS
from schema import Transaction, User

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT_SECONDS})
Session = sessionmaker(bind=engine)

CANONICAL = set(CATEGORIES) | {"Uncategorized"}

DATE_ALIASES = (
    "transaction date",
    "transaction_date",
    "clearing date",
    "clearing_date",
    "date",
    "posting_date",
    "posting date",
    "trans_date",
    "trans date",
    "posted",
    "post_date",
    "post date",
    "value_date",
    "value date",
)
AMOUNT_ALIASES = (
    "amount (usd)",
    "amount(usd)",
    "amount",
    "amt",
    "debit",
    "outflow",
    "spend",
    "withdrawal",
)
# Prefer "category" before "type" so Debit/Credit "Type" columns are not used when Category exists.
CATEGORY_ALIASES = (
    "category",
    "category_name",
    "category name",
    "expense_category",
    "expense category",
    "class",
    "type",
)
MERCHANT_ALIASES = (
    "merchant",
    "merchant_name",
    "description",
    "payee",
    "memo",
    "name",
    "narrative",
)

# Normalized slug / keyword -> pipeline category
SLUG_MAP = {
    "food_dining": "Food and Drink",
    "food_and_drink": "Food and Drink",
    "dining": "Food and Drink",
    "restaurants": "Food and Drink",
    "restaurant": "Food and Drink",
    "groceries": "Groceries",
    "grocery": "Groceries",
    "supermarket": "Groceries",
    "shopping": "Shopping",
    "retail": "Shopping",
    "travel": "Travel",
    "transportation": "Travel",
    "gas": "Travel",
    "fuel": "Travel",
    "auto": "Travel",
    "entertainment": "Entertainment",
    "health": "Health",
    "medical": "Health",
    "pharmacy": "Health",
    "fitness": "Health",
    "utilities": "Utilities",
    "utility": "Utilities",
    "bills": "Utilities",
    "home": "Utilities",
    "misc": "Uncategorized",
    "other": "Uncategorized",
    "uncategorized": "Uncategorized",
}


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).lower().strip(): c for c in df.columns}


def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    lookup = _column_lookup(df)
    for a in aliases:
        if a in lookup:
            return lookup[a]
    return None


def resolve_category(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Uncategorized"
    s = str(raw).strip()
    if not s:
        return "Uncategorized"
    if s in CANONICAL:
        return s
    low = s.lower()
    slug = (
        low.replace(" ", "_")
        .replace("&", "and")
        .replace("-", "_")
        .replace("/", "_")
    )
    if slug in SLUG_MAP:
        return SLUG_MAP[slug]
    # substring hints
    if "grocery" in low or "supermarket" in low:
        return "Groceries"
    if "restaurant" in low or "dining" in low or "food" in low:
        return "Food and Drink"
    if "gas" in low or "fuel" in low or "uber" in low or "lyft" in low:
        return "Travel"
    if "entertainment" in low or "streaming" in low or "movie" in low:
        return "Entertainment"
    if "utility" in low or "electric" in low or "internet" in low:
        return "Utilities"
    if "health" in low or "medical" in low or "pharmacy" in low:
        return "Health"
    if "shop" in low or "retail" in low or "amazon" in low:
        return "Shopping"
    return "Uncategorized"


def _normalize_amounts(series: pd.Series, debits_negative: bool) -> pd.Series:
    """Convert CSV amount column to positive spend."""
    s = pd.to_numeric(series, errors="coerce")
    if not debits_negative:
        return s
    # Bank checking: outflows often negative. Corporate cards: charges often all positive.
    if (s < 0).any():
        spend = s.where(s < 0, other=pd.NA)
        return -spend
    return s


def read_statement_dataframe(
    source: CsvSource,
    *,
    debits_negative: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("CSV is empty.")

    date_col = _pick_column(df, DATE_ALIASES)
    amount_col = _pick_column(df, AMOUNT_ALIASES)
    if not date_col or not amount_col:
        raise ValueError(
            f"Could not find date and/or amount columns. "
            f"Columns found: {list(df.columns)}. "
            f"Use one of date={DATE_ALIASES}, amount={AMOUNT_ALIASES}."
        )

    cat_col = _pick_column(df, CATEGORY_ALIASES)
    merch_col = _pick_column(df, MERCHANT_ALIASES)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    out["amount"] = _normalize_amounts(df[amount_col], debits_negative=debits_negative)
    if cat_col:
        out["category"] = df[cat_col].map(resolve_category)
    else:
        out["category"] = "Uncategorized"
    if merch_col:
        out["merchant_name"] = df[merch_col].astype(str).str.slice(0, 100)
    else:
        out["merchant_name"] = "Unknown"

    bad = out["date"].isna() | out["amount"].isna()
    out = out.loc[~bad].copy()
    out = out[out["amount"] > 0]
    return out


def ensure_user(session, user_id: int, username: Optional[str] = None) -> None:
    user = session.query(User).filter_by(id=user_id).first()
    if user:
        return
    session.add(
        User(
            id=user_id,
            username=username or f"user_{user_id}",
            email=f"user_{user_id}@import.local",
        )
    )
    session.commit()


def import_statement_csv(
    source: Union[str, bytes, BinaryIO],
    user_id: int,
    *,
    debits_negative: bool = True,
    account_id: str = "csv_import",
    ensure_user_exists: bool = True,
) -> dict:
    """
    Parse CSV and insert transactions for user_id.

    `source` can be a file path, or bytes / file-like (e.g. UploadFile.file).
    """
    if isinstance(source, bytes):
        buf: Union[io.BytesIO, str] = io.BytesIO(source)
    else:
        buf = source

    df = read_statement_dataframe(buf, debits_negative=debits_negative)
    if df.empty:
        return {"inserted": 0, "skipped": 0, "message": "No valid rows after parsing."}

    session = Session()
    try:
        if ensure_user_exists:
            ensure_user(session, user_id)

        rows = [
            Transaction(
                user_id=user_id,
                date=row["date"],
                amount=float(row["amount"]),
                category=row["category"],
                merchant_name=str(row.get("merchant_name", "Unknown"))[:100],
                account_id=account_id,
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(rows)
        session.commit()
        return {
            "inserted": len(rows),
            "skipped": 0,
            "message": None,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a statement CSV into PostgreSQL.")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--user-id", type=int, required=True, help="Target users.id")
    parser.add_argument(
        "--credits-positive",
        action="store_true",
        help="Spending amounts are already positive in CSV; negative-only debit parsing is disabled.",
    )
    parser.add_argument(
        "--account-id",
        default="csv_import",
        help="Stored in transactions.account_id (default: csv_import)",
    )
    args = parser.parse_args()

    debits_negative = not args.credits_positive
    result = import_statement_csv(
        args.csv_path,
        args.user_id,
        debits_negative=debits_negative,
        account_id=args.account_id,
    )
    print(f"Inserted {result['inserted']} transactions for user_id={args.user_id}")
    if result.get("message"):
        print(result["message"])
    print("Next: run model/train.py for this user, then call GET /anomaly/check/{user_id}")


if __name__ == "__main__":
    main()
