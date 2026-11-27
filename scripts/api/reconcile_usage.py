"""Inspect retained ledger evidence without modifying historical records."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from superagent.governance.usage_ledger import UsageLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger-dir', type=Path, required=True)
    parser.add_argument('--evaluation-id', required=True)
    parser.add_argument('--budget-rmb', default='20')
    parser.add_argument('--evaluation-stage', choices=['development', 'released', 'unseen'], required=True)
    parser.add_argument('--apply', action='store_true', help='Deprecated: historical writes are prohibited')
    args = parser.parse_args()
    if args.apply:
        parser.error('Historical records are read-only; upper-bound reconciliation is no longer required')
    if not (args.ledger_dir / 'ledger.json').is_file():
        parser.error('Inspection requires an existing ledger; no new ledger directory is created')
    ledger = UsageLedger(args.ledger_dir, args.evaluation_id, args.budget_rmb, args.evaluation_stage)
    print(json.dumps({'dry_run': True, 'summary': ledger.summary(), 'original_records': ledger.audit_records()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
