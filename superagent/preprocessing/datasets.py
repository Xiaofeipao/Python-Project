"""Data inspection and validation without reading the designated test set."""

import hashlib
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path, id_column=None, target_column=None):
    # String IDs/labels retain leading zeroes and stay stable across CSV files.
    dtypes = {column: "string" for column in (id_column, target_column) if column}
    frame = pd.read_csv(path, dtype=dtypes or None)
    if not len(frame):
        raise ValueError(f"Empty input CSV: {path}")
    if id_column:
        if id_column not in frame or frame[id_column].isna().any() or not frame[id_column].is_unique:
            raise ValueError(f"ID column {id_column!r} must exist, be nonempty and unique: {path}")
    return frame


def labels(frame, target):
    if target not in frame:
        raise ValueError(f"Missing target column {target!r}")
    if frame[target].isna().any():
        raise ValueError("Target labels must not be missing")
    return frame[target].astype(str)


def text_score(series):
    """Modality evidence from content, never from a column or dataset name."""
    values = series.dropna().astype(str)
    if not len(values):
        return {"non_numeric": 0.0, "mean_words": 0.0, "median_length": 0.0, "unique_ratio": 0.0}
    numeric = pd.to_numeric(values, errors="coerce").notna().mean()
    words = values.str.split().str.len()
    lengths = values.str.len()
    return {"non_numeric": float(1 - numeric), "mean_words": float(words.mean()),
            "median_length": float(lengths.median()),
            "unique_ratio": float(values.nunique() / len(values))}


def is_text_like(score):
    # Short paired texts (3+ words) qualify; one-word categories, codes and IDs do not.
    if score["non_numeric"] < 0.9 or score["unique_ratio"] < 0.05:
        return False
    return score["mean_words"] >= 2.5 or score["median_length"] >= 40


def is_datetime_like(series):
    values = series.dropna()
    if not len(values):
        return False
    parsed = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.9:
        return False
    years = parsed.dt.year.dropna()
    return bool(len(years)) and 1900 <= years.min() and years.max() <= 2100


def inspect_data(config):
    frame = read_csv(config.train_csv, config.id_column, config.target_column)
    y = labels(frame, config.target_column)
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two classes")
    excluded = {config.target_column, config.id_column}
    all_features = config.feature_columns or [c for c in frame if c not in excluded]
    if not all_features or any(c in excluded or c not in frame for c in all_features):
        raise ValueError("Feature columns must exist and exclude target/ID")
    modality = config.data_type
    text_columns = config.text_columns
    if modality == "auto":
        if not config.image_column:
            image_candidates = [c for c in all_features if not is_numeric_dtype(frame[c])
                and frame[c].dropna().astype(str).str.lower().str.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')).mean() > .9]
            if len(image_candidates) == 1:
                config.image_column = image_candidates[0]
        if config.image_column:
            modality = "image"
        elif text_columns:
            modality = "text"
        else:
            scores = {c: text_score(frame[c]) for c in all_features if not is_numeric_dtype(frame[c])}
            detected = [c for c, value in scores.items() if is_text_like(value)]
            modality = "text" if detected else "tabular"
            text_columns = detected or None
    if modality == "text":
        text_columns = text_columns or [c for c in all_features if not is_numeric_dtype(frame[c])]
        if not text_columns or any(c not in all_features for c in text_columns):
            raise ValueError("Text columns must be valid non-target, non-ID feature columns")
    if modality == "image":
        if not config.image_column or config.image_column not in all_features:
            raise ValueError("image_column must name a valid non-target, non-ID feature column")
    tabular_columns = [c for c in all_features
                       if c not in set(text_columns or []) and c != config.image_column]
    numeric = [c for c in tabular_columns if is_numeric_dtype(frame[c])]
    datetimes = [c for c in tabular_columns if c not in numeric and is_datetime_like(frame[c])]
    categorical = [c for c in tabular_columns if c not in numeric and c not in datetimes]
    profile = {
        "modality_evidence": {"image_column": config.image_column, "text_columns": text_columns,
                              "rule": "Explicit columns, then image extensions (>90%), then text-likeness "
                                      "(non-numeric >90%, unique ratio >=5%, mean words >=2.5 or median length >=40)"},
        "rows": len(frame), "features": all_features, "data_type": modality,
        "text_columns": text_columns, "image_column": config.image_column,
        "tabular_columns": tabular_columns,
        "numeric_columns": numeric, "categorical_columns": categorical,
        "datetime_columns": datetimes,
        "categorical_cardinality": {c: int(frame[c].nunique(dropna=True)) for c in categorical},
        "text_likeness": {c: text_score(frame[c]) for c in all_features
                          if not is_numeric_dtype(frame[c]) and c not in set(text_columns or [])},
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "missing_fractions": {c: float(frame[c].isna().mean()) for c in all_features},
    }
    if config.positive_label is not None and config.positive_label not in set(y):
        raise ValueError("positive_label does not appear in training labels")
    if config.positive_label is not None and y.nunique() != 2:
        raise ValueError("positive_label applies to binary classification only")
    hashes = {"train_csv": file_hash(config.train_csv)}
    if config.validation_csv:
        validation = read_csv(config.validation_csv, config.id_column, config.target_column)
        hashes["validation_csv"] = file_hash(config.validation_csv)
        if hashes["validation_csv"] == hashes["train_csv"]:
            raise ValueError("Training and validation files have identical content")
        missing = set(all_features) - set(validation)
        if missing:
            raise ValueError(f"Validation CSV missing columns: {sorted(missing)}")
        vy = labels(validation, config.target_column)
        if set(vy) - set(y):
            raise ValueError("Validation contains classes absent from training")
        if config.id_column and set(frame[config.id_column]) & set(validation[config.id_column]):
            raise ValueError("Train and validation IDs overlap")
        train = frame.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        split = {"kind": "provided", "train_indices": list(range(len(train))),
                 "validation_indices": list(range(len(validation)))}
    else:
        try:
            classes = y.nunique()
            fraction = max(config.validation_fraction, classes / len(frame))
            fraction = min(fraction, 1 - classes / len(frame))
            train_idx, val_idx = train_test_split(
                list(range(len(frame))), test_size=fraction,
                stratify=y, random_state=config.seed,
            )
        except ValueError as exc:
            # Singleton classes stay in training. Validation cannot measure their
            # recall, so record this limitation rather than dropping the class.
            train_idx, val_idx = [], []
            for label in sorted(set(y)):
                indices = list(frame.index[y == label])
                if len(indices) >= 2:
                    tr, va = train_test_split(indices, test_size=min(len(indices)-1,
                        max(1, int(len(indices)*config.validation_fraction))), random_state=config.seed)
                    train_idx.extend(tr)
                    val_idx.extend(va)
                else:
                    train_idx.extend(indices)
            if not val_idx:
                raise ValueError('No class has enough rows for validation; provide validation_csv') from exc
            profile['validation_warning'] = 'Rare-class fallback: singleton classes remain in training; validation may omit classes'
        train = frame.iloc[train_idx].reset_index(drop=True)
        validation = frame.iloc[val_idx].reset_index(drop=True)
        split = {"kind": "stratified_holdout", "train_indices": train_idx, "validation_indices": val_idx}
    if config.metric in {"roc_auc", "log_loss"}:
        if set(labels(validation, config.target_column)) != set(labels(train, config.target_column)):
            raise ValueError("AUC/log_loss require all training classes in validation")
    # Test path is deliberately not opened, hashed or profiled.
    return train, validation, profile, split, hashes


def image_hashes(frames, column, root):
    """Hash only images used for fitting/validation, never designated test images."""
    records = {}
    for frame in frames:
        for value in frame[column]:
            if pd.isna(value) or not str(value).strip():
                raise ValueError("Image paths must not be missing")
            path = (Path(root) / str(value)).resolve()
            if str(path) not in records:
                records[str(path)] = file_hash(path)
    return records
