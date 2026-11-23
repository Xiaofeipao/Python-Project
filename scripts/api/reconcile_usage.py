"""Reconcile unrecorded model calls in one task ledger without inventing usage.

A read timeout proves the request was sent but not that it generated nothing.
This tool never writes token counts or a zero-cost success. With ``--apply`` it
books the call's worst-case reservation as an explicit conservative upper bound
and records the basis plus the original evidence path, leaving
``accounting_complete`` False so the run stays ineligible for formal submission.

Dry run is the default; it lists unrecorded calls and the current summary.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from superagent.governance.usage_ledger import UsageLedger


def _looks_like_path(value):
    return value.startswith("/") or (len(value) > 2 and value[1] == ":" and value[2] in "\\/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True,
                        help="Ledger directory .model_usage/<sha256(evaluation_id)>")
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--budget-rmb", default="20")
    parser.add_argument("--call-id")
    parser.add_argument("--basis", help="Why the usage cannot be recovered and how it is charged")
    parser.add_argument("--evidence", help="Original request path or record id proving the failure")
    parser.add_argument("--reconciled-by", default="usage-reconciliation")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    ledger = UsageLedger(args.ledger_dir, args.evaluation_id, args.budget_rmb)
    state = ledger.audit_records()
    unrecorded = [c for c in state["calls"] if c["status"] != "recorded"]
    report = {"dry_run": not args.apply, "evaluation_id": args.evaluation_id,
              "ledger_path": str(ledger.path), "summary_before": ledger.summary(),
              "unrecorded_calls": [{"call_id": c["call_id"], "status": c["status"],
                                    "error": c.get("error"), "reserved_rmb": c.get("reserved_rmb"),
                                    "cost_rmb": c.get("cost_rmb"), "timestamp_utc": c.get("timestamp_utc"),
                                    "request_path": c.get("request_path")} for c in unrecorded]}

    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if not (args.call_id and args.basis and args.evidence):
        parser.error("--apply requires --call-id, --basis and --evidence")
    if _looks_like_path(args.evidence) and not Path(args.evidence).exists():
        parser.error(f"Evidence path does not exist: {args.evidence}")

    call = ledger.reconcile_unknown(args.call_id, args.basis, args.evidence, args.reconciled_by)
    report["reconciliation"] = call["reconciliation"]
    report["charged_rmb"] = call["cost_rmb"]
    report["summary_after"] = ledger.summary()

    audit_path = ledger.directory / "reconciliation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {
        "schema_version": 1, "evaluation_id": args.evaluation_id, "records": []}
    audit["records"].append({"call_id": call["call_id"], "error": call.get("error"),
                             "conservative_charge_rmb": call["cost_rmb"],
                             **call["reconciliation"],
                             "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report["audit_path"] = str(audit_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
