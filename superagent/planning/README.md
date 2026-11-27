# Planning Responsibilities

`llm.py` owns the single planner's feedback summaries, supported API interaction,
reply parsing and deterministic fallback. `proposals.py` validates permissible
changes and rejects duplicates; `memory.py` summarizes previous trials.

The `search/` package separates two parts of local planning:

- `candidates.py`: complete generic candidate menu, initial ordering and refinement.
- `allocation.py`: candidate identity, metric-specific variants and resource estimates.

Estimator construction is in `../models/factory.py`, not the planner. The older
`search_space.py` import path re-exports these implementations for compatibility.
Existing candidate values, ordering, budget limits and model selection semantics
are preserved.
