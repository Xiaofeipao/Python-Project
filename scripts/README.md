# Script Roles

- `workflows/`: batch evaluation and bounded automatic recovery.
- `data/`: trusted NumPy task conversion, preserving supplied partitions.
- `api/`: connectivity probes and explicit ledger reconciliation.
- `validation/`: development benchmarks, artifact checks and frozen-workflow checks.
- `reporting/`: timelines generated from actual recorded events.

Run from the repository root. Use these organized paths rather than copying
implementations into root scripts. Each standalone implementation establishes
the repository import path at most once when it needs project imports.

The released batch workflow runs one shared development policy serially and
never freezes implicitly. Freezing and frozen-workflow verification are explicit
separate operations. Reporting and final evaluation scripts read results only;
their test scores must not guide subsequent model selection.

Probes may incur API charges. Reconciliation is read-only by default and requires explicit evidence for changes. None of these scripts should be run merely to browse the accounting inventory or experiment catalog.

`setup/download_models.py` prepares all pinned text snapshots/tokenizers and
ImageNet weights, or checks local resources with `--offline`. It does not train
or invoke the planner API. See `docs/guides/model_cache.md` for paths and copying.
