# Script Roles

- `workflows/`: batch evaluation and bounded automatic recovery.
- `data/`: trusted NumPy task conversion, preserving supplied partitions.
- `api/`: connectivity probes and explicit ledger reconciliation.
- `validation/`: development benchmarks, artifact checks and frozen-workflow checks.
- `reporting/`: timelines generated from actual recorded events.

Run from the repository root. New instructions use these organized paths. Root-level scripts are compatibility forwarders, not duplicate implementations. Each standalone implementation establishes the repository import path at most once when it needs project imports.

Probes may incur API charges. Reconciliation is read-only by default and requires explicit evidence for changes. None of these scripts should be run merely to browse the accounting inventory or experiment catalog.
