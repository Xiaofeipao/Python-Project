# Model Implementations

Start with `factory.py`: it builds the requested estimator and feature pipeline
without deciding the next experiment or reading test data.

- `tabular.py`: native categorical adapters, tree/boosting estimators and blends.
- `neural.py`: shared numeric/image neural training, ensembles and frozen images.
- `decision.py`: fixed decision thresholds and training-fold calibration.
- `text/encoders.py`: pretrained encoder registry, pinned downloads and features.
- `text/networks.py`: trainable EmbeddingBag, CNN and GRU text classifiers.
- `text/finetuning.py`: optional end-to-end Transformer fine-tuning.
- `text/pairwise.py`: composition of ordered-pair overlap classifiers.

The older `text_encoder.py`, `text_neural.py`, `text_finetune.py` and `pairwise.py`
paths are import/pickle compatibility facades. Edit their canonical counterparts
above. Historical models require compatible dependencies and their recorded
resources; retaining an import path does not bypass source verification.
