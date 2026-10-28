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


def inspect_data(config):
    frame = read_csv(config.train_csv, config.id_column, config.target_column)
    y = labels(frame, config.target_column)
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two classes")
    excluded = {config.target_column, config.id_column}
    features = config.feature_columns or [c for c in frame if c not in excluded]
    if not features or any(c in excluded or c not in frame for c in features):
        raise ValueError("Feature columns must exist and exclude target/ID")
    modality = config.data_type
    text_columns = config.text_columns
    if modality == "auto":
        if config.image_column:
            modality = "image"
        elif text_columns:
            modality = "text"
        else:
            long_text = [c for c in features if not is_numeric_dtype(frame[c])
                         and frame[c].dropna().astype(str).str.len().median() > 80]
            modality = "text" if long_text else "tabular"
            text_columns = long_text or None
    if modality == "text":
        text_columns = text_columns or [c for c in features if not is_numeric_dtype(frame[c])]
        if not text_columns or any(c not in features for c in text_columns):
            raise ValueError("Text columns must be valid non-target, non-ID feature columns")
        features = text_columns
    if modality == "image":
        if not config.image_column or config.image_column not in features:
            raise ValueError("image_column must name a valid non-target, non-ID feature column")
        features = [config.image_column]
    numeric = [c for c in features if is_numeric_dtype(frame[c])]
    profile = {
        "rows": len(frame), "features": features, "data_type": modality,
        "numeric_columns": numeric, "categorical_columns": [c for c in features if c not in numeric],
        "text_columns": text_columns, "image_column": config.image_column,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "missing_fractions": {c: float(frame[c].isna().mean()) for c in features},
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
        missing = set(features) - set(validation)
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
        if y.value_counts().min() < 2:
            raise ValueError("Each class needs at least two rows for a stratified validation split")
        try:
            train_idx, val_idx = train_test_split(
                list(range(len(frame))), test_size=config.validation_fraction,
                stratify=y, random_state=config.seed,
            )
        except ValueError as exc:
            raise ValueError(f"Cannot make stratified split; adjust fraction or provide validation_csv: {exc}") from exc
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
