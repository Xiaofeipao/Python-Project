"""Optional end-to-end fine-tuning of a frozen text encoder.

This is intentionally a separate, opt-in family: it is much more expensive
than a linear probe and should only be exposed when a GPU and a cached
checkpoint are available.  The fitting frame is the only source of
parameters; the validation split is used by the orchestrator, and test rows
never enter this estimator.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from superagent.models.text.encoders import encoder_spec, _download_snapshot


class TransformerFinetuneClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, columns, encoder='mpnet', epochs=3, learning_rate=2e-5,
                 weight_decay=0.01, batch_size=16, max_length=256, device='auto',
                 threads=1, random_state=42, freeze_layers=0):
        self.columns = columns
        self.encoder = encoder
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.threads = threads
        self.random_state = random_state
        self.freeze_layers = freeze_layers

    def __setstate__(self, state):
        self.__dict__.update(type(self)(columns=[]).get_params(deep=False))
        self.__dict__.update(state)

    def _device(self):
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        return 'cuda' if self.device != 'cpu' and torch.cuda.is_available() else 'cpu'

    def _documents(self, X):
        return X[self.columns].fillna('').astype(str).apply(
            lambda row: ' [SEP] '.join(str(value) for value in row), axis=1).tolist()

    def _load(self, labels=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = encoder_spec(self.encoder)
        path = _download_snapshot(spec)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=labels, local_files_only=True, trust_remote_code=False)
        return path, tokenizer, model

    def fit(self, X, y):
        import torch
        from torch import nn

        device = self._device()
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Fine-tuning requires at least two classes')
        path, tokenizer, model = self._load(len(self.classes_))
        self.model_path_ = path
        model.to(device)
        if self.freeze_layers:
            # Freeze the embedding/pooler stack while keeping encoder blocks
            # trainable; the exact module names differ by architecture, so
            # only freeze the shared embedding namespace.
            for name, parameter in model.named_parameters():
                if name.startswith('bert.embeddings') or name.startswith('roberta.embeddings') \
                        or name.startswith('mpnet.embeddings') or name.startswith('deberta.embeddings'):
                    parameter.requires_grad = False
        documents = self._documents(X)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        generator = np.random.default_rng(self.random_state)
        targets = torch.tensor(encoded, dtype=torch.long, device=device)
        losses = []
        model.train()
        for _ in range(max(1, int(self.epochs))):
            order = generator.permutation(len(documents))
            total = torch.zeros((), device=device)
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                tokenized = tokenizer([documents[index] for index in indices],
                                      padding=True, truncation=True, max_length=self.max_length,
                                      return_tensors='pt')
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                target = targets[torch.tensor(indices, device=device)]
                optimizer.zero_grad(set_to_none=True)
                logits = model(**tokenized).logits
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.detach().float() * len(indices)
            value = float(total / len(documents))
            if not np.isfinite(value):
                raise ValueError('Fine-tuning loss became nonfinite')
            losses.append(value)
        self.weights_ = {key: value.detach().cpu().numpy().copy()
                         for key, value in model.state_dict().items()}
        self.training_report_ = {
            'device': device, 'encoder': self.encoder, 'epochs': len(losses),
            'model_id': encoder_spec(self.encoder)['model_id'],
            'revision': encoder_spec(self.encoder)['revision'],
            'training_loss': losses, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'gpu_name': torch.cuda.get_device_name() if device == 'cuda' else None,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
        }
        model.to('cpu')
        if device == 'cuda':
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, ['weights_', 'classes_', 'model_path_'])
        device = self._device()
        _, tokenizer, model = self._load(len(self.classes_))
        model.load_state_dict({key: torch.from_numpy(value).to(device)
                               for key, value in self.weights_.items()})
        model.to(device)
        model.eval()
        documents = self._documents(X)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                tokenized = tokenizer(documents[start:start + self.batch_size],
                                      padding=True, truncation=True, max_length=self.max_length,
                                      return_tensors='pt')
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                batches.append(torch.softmax(model(**tokenized).logits.float(), dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
