"""
Run the full pipeline from a CSV file on disk (no HTTP upload).

From the project root:

  python -m ingest.local_report path/to/statement.csv --user-id 1

Optional:
  --credits-positive     same as CSV importer (positive = spend)
  --skip-train           only ingest + anomaly (use existing S3 models)
  --output report.json   write JSON report to a file instead of stdout

Next: open report.json or pipe to a viewer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUDGETS, CATEGORIES
from ingest.csv_statement import import_statement_csv
from model.anomaly import run_anomaly_check
from model.train import train_all_models_for_user


def build_report(
    csv_path: str,
    user_id: int,
    *,
    debits_negative: bool,
    account_id: str,
    retrain: bool,
) -> dict:
    with open(csv_path, "rb") as f:
        raw = f.read()
    if not raw:
        raise ValueError("CSV file is empty.")

    ingest_result = import_statement_csv(
        raw,
        user_id,
        debits_negative=debits_negative,
        account_id=account_id,
    )

    if retrain:
        train_out = train_all_models_for_user(user_id, quiet=True)
    else:
        from model.train import load_transactions

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
            "message": "Retrain skipped (--skip-train).",
        }

    payload = run_anomaly_check(user_id, CATEGORIES, BUDGETS, request_id="local-report")
    results = payload["results"]
    err_list = payload["errors"]

    flagged_count = sum(1 for r in results if r.get("flag"))
    ok_count = sum(1 for r in results if not r.get("flag"))
    alerts = [r["alert"] for r in results if r.get("alert")]

    return {
        "user_id": user_id,
        "csv_path": os.path.abspath(csv_path),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a local CSV, train models, run anomaly detection, emit JSON report."
    )
    parser.add_argument("csv_path", help="Path to statement CSV on disk")
    parser.add_argument("--user-id", type=int, required=True, help="users.id")
    parser.add_argument(
        "--credits-positive",
        action="store_true",
        help="Spending amounts are already positive (see ingest.csv_statement).",
    )
    parser.add_argument(
        "--account-id",
        default="csv_import",
        help="transactions.account_id (default: csv_import)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not retrain; use existing S3 models after ingest.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Write JSON to this file instead of printing to stdout",
    )
    args = parser.parse_args()

    path = args.csv_path
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    debits_negative = not args.credits_positive
    report = build_report(
        path,
        args.user_id,
        debits_negative=debits_negative,
        account_id=args.account_id,
        retrain=not args.skip_train,
    )
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
