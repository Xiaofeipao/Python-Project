# Feature Preparation

`datasets.py` loads and inspects inputs, preserves official splits and records
input hashes. `pipelines.py` composes feature components for the requested
modality. `feature_selection.py` implements optional train-fitted filters.

Reusable serializable components live in `components/`:

- `base.py`: constant baseline features and bounded SVD.
- `tabular.py`: schema enforcement, datetime expansion and frequency encoding.
- `text.py`: safe TF-IDF, joined/separate text and signed/overlap pair features.
- `image.py`: deterministic image reading, resized pixels and color histograms.

`transformers.py` preserves historical imports and pickle lookups. New code
should import components or pipeline builders directly. Fitting statistics,
vocabularies and filters still use the fitting partition only.
