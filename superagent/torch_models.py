"""Optional GPU learners: batching, AMP, train-only stopping and seed ensembles."""

from contextlib import nullcontext

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class TorchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, architecture="mlp", hidden=128, epochs=40, learning_rate=0.001,
                 weight_decay=0.001, batch_size=128, image_size=32, device="auto",
                 random_state=42, threads=1, depth=3, width=32, dropout=0.1,
                 mixed_precision=False, gpu_cache_fraction=0.2, early_stopping=False,
                 patience=12, label_smoothing=0.0, augment=False, cosine_schedule=False,
                 refit_epochs=None, normalize_images=False):
        self.architecture = architecture
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.random_state = random_state
        self.threads = threads
        self.depth = depth
        self.width = width
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.gpu_cache_fraction = gpu_cache_fraction
        self.early_stopping = early_stopping
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.augment = augment
        self.cosine_schedule = cosine_schedule
        self.refit_epochs = refit_epochs
        self.normalize_images = normalize_images

    def __setstate__(self, state):
        # Earlier checkpoints lack the new parameters; preserve legacy defaults.
        self.__dict__.update(type(self)().get_params(deep=False))
        self.__dict__.update(state)

    def _device(self):
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but unavailable")
        return "cuda" if self.device != "cpu" and torch.cuda.is_available() else "cpu"

    def _network(self):
        from torch import nn

        class ResidualConv(nn.Module):
            def __init__(self, incoming, outgoing, stride=1):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing), nn.ReLU(),
                    nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.BatchNorm2d(outgoing),
                )
                self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
                    nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False), nn.BatchNorm2d(outgoing))
                self.activation = nn.ReLU()

            def forward(self, value):
                return self.activation(self.layers(value) + self.skip(value))

        class ResidualDense(nn.Module):
            def __init__(self, hidden, dropout):
                super().__init__()
                self.layers = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2),
                                            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))

            def forward(self, value):
                return value + self.layers(value)

        if self.architecture == "resnet":
            width = self.width
            return nn.Sequential(
                nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                ResidualConv(width, width), ResidualConv(width, width * 2, stride=2),
                ResidualConv(width * 2, width * 2), ResidualConv(width * 2, width * 4, stride=2),
                nn.AvgPool2d(self.image_size // 4), nn.Flatten(),
                nn.Linear(width * 4, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(self.hidden, len(self.classes_)),
            )
        if self.architecture == "cnn":
            return nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * (self.image_size // 4) ** 2, self.hidden), nn.ReLU(),
                nn.Dropout(self.dropout), nn.Linear(self.hidden, len(self.classes_)),
            )
        if self.architecture == "residual_mlp":
            return nn.Sequential(
                nn.Linear(self.n_features_in_, self.hidden), nn.GELU(),
                *[ResidualDense(self.hidden, self.dropout) for _ in range(self.depth)],
                nn.LayerNorm(self.hidden), nn.Linear(self.hidden, len(self.classes_)),
            )
        return nn.Sequential(
            nn.Linear(self.n_features_in_, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2), nn.ReLU(),
            nn.Linear(self.hidden // 2, len(self.classes_)),
        )

    def _batch(self, X, indices, device):
        import torch

        batch = X[indices]
        if sparse.issparse(batch):
            batch = batch.toarray()
        batch = np.asarray(batch, dtype=np.float32)
        if self.architecture in {"cnn", "resnet"}:
            batch = batch[:, :self.image_size * self.image_size * 3]
            batch = batch.reshape(-1, self.image_size, self.image_size, 3).transpose(0, 3, 1, 2)
            if self.normalize_images:
                batch = (batch - 0.5) * 2
        return torch.from_numpy(np.ascontiguousarray(batch)).to(device)

    def _precision(self, device):
        import torch

        if device == "cuda" and self.mixed_precision:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None

    def _autocast(self, device, dtype):
        import torch

        return torch.autocast(device_type=device, dtype=dtype) if dtype is not None else nullcontext()

    def fit(self, X, y):
        import torch
        from torch import nn

        device = self._device()
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        self.n_features_in_ = X.shape[1]
        if self.architecture not in {"cnn", "resnet"} and self.n_features_in_ * self.hidden > 20_000_000:
            raise ValueError("Neural input layer exceeds the implementation size limit")
        training = np.arange(len(encoded))
        monitor = np.array([], dtype=int)
        use_stopping = self.early_stopping and self.refit_epochs is None
        if use_stopping and len(encoded) >= max(50, len(self.classes_) * 10) and np.bincount(encoded).min() >= 3:
            training, monitor = train_test_split(training, test_size=max(0.15, len(self.classes_) / len(encoded)),
                                                  stratify=encoded, random_state=self.random_state)
        maximum_epochs = int(self.refit_epochs or self.epochs)
        model = self._network().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs) if self.cosine_schedule else None
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        dtype = self._precision(device)
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and dtype == torch.float16)
        generator = np.random.default_rng(self.random_state)
        losses, monitor_losses = [], []
        best_weights, best_loss, best_epoch, stale = None, float("inf"), 0, 0
        cached = None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            required_bytes = int(X.shape[0]) * int(X.shape[1]) * 4
            free_bytes = torch.cuda.mem_get_info()[0]
            if required_bytes < min(free_bytes * self.gpu_cache_fraction, 2 * 1024**3):
                cached = self._batch(X, slice(None), device)
        target_cache = torch.as_tensor(encoded, dtype=torch.long, device=device)

        def features(indices):
            return cached[torch.as_tensor(indices, device=device)] if cached is not None else self._batch(X, indices, device)

        for epoch in range(maximum_epochs):
            model.train()
            order = generator.permutation(training)
            accumulated = torch.zeros((), device=device)
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = features(indices)
                if self.augment and self.architecture in {"cnn", "resnet"}:
                    shift = max(1, self.image_size // 16)
                    dy, dx = generator.integers(-shift, shift + 1, size=2).tolist()
                    padded = nn.functional.pad(batch, (shift, shift, shift, shift))
                    batch = padded[:, :, shift + dy:shift + dy + self.image_size,
                                   shift + dx:shift + dx + self.image_size]
                target = target_cache[torch.as_tensor(indices, device=device)]
                optimizer.zero_grad(set_to_none=True)
                with self._autocast(device, dtype):
                    loss = criterion(model(batch), target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                accumulated += loss.detach().float() * len(indices)
            epoch_loss = float(accumulated / len(training))
            if not np.isfinite(epoch_loss):
                raise ValueError("Neural training loss became nonfinite")
            losses.append(epoch_loss)
            if scheduler:
                scheduler.step()
            if len(monitor):
                model.eval()
                total = torch.zeros((), device=device)
                with torch.inference_mode():
                    for start in range(0, len(monitor), self.batch_size):
                        indices = monitor[start:start + self.batch_size]
                        with self._autocast(device, dtype):
                            logits = model(features(indices))
                            value = nn.functional.cross_entropy(logits.float(), target_cache[torch.as_tensor(indices, device=device)])
                        total += value * len(indices)
                measured = float(total / len(monitor))
                if not np.isfinite(measured):
                    raise ValueError("Internal stopping loss became nonfinite")
                monitor_losses.append(measured)
                if measured < best_loss - 1e-4:
                    best_loss, best_epoch, stale = measured, epoch + 1, 0
                    best_weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        self.weights_ = best_weights if best_weights is not None else {
            k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        self.training_report_ = {
            "device": device, "epochs": len(losses), "best_epoch": best_epoch or len(losses),
            "training_loss": losses, "internal_monitor_loss": monitor_losses,
            "internal_monitor_rows": len(monitor), "training_rows": len(training),
            "precision": str(dtype) if dtype is not None else "float32",
            "data_placement": "gpu_cached" if cached is not None else "streamed_batches",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        }
        return self

    def predict_proba(self, X):
        import torch

        check_is_fitted(self, ["weights_", "classes_"])
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from fitted schema")
        device = self._device()
        model = self._network().to(device)
        model.load_state_dict({key: torch.from_numpy(value).to(device) for key, value in self.weights_.items()})
        model.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, X.shape[0], self.batch_size):
                logits = model(self._batch(X, slice(start, start + self.batch_size), device))
                batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchEnsemble(ClassifierMixin, BaseEstimator):
    """Seed averaging; members see only allowed fitting rows."""

    def __init__(self, member_params=None, size=3, random_state=42, device="auto", refit_epochs=None):
        self.member_params = member_params
        self.size = size
        self.random_state = random_state
        self.device = device
        self.refit_epochs = refit_epochs

    def fit(self, X, y):
        self.estimators_ = []
        for index in range(self.size):
            params = dict(self.member_params or {})
            epochs = self.refit_epochs[index] if self.refit_epochs is not None else None
            member = TorchClassifier(**dict(params, random_state=(self.random_state + index * 1009) % 2**32,
                                            device=self.device, refit_epochs=epochs))
            member.fit(X, y)
            self.estimators_.append(member)
        self.classes_ = self.estimators_[0].classes_
        self.n_features_in_ = X.shape[1]
        members = [member.training_report_ for member in self.estimators_]
        self.training_report_ = {
            "device": members[0]["device"], "members": members,
            "best_epoch": [member["best_epoch"] for member in members],
            "peak_cuda_allocated_bytes": max(member["peak_cuda_allocated_bytes"] for member in members),
        }
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        probabilities = []
        for member in self.estimators_:
            member.set_params(device=self.device)
            probabilities.append(member.predict_proba(X))
        return np.mean(probabilities, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
