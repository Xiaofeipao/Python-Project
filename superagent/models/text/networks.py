"""Trainable text networks that do not rely on a pretrained checkpoint.

These are deliberately small, deterministic baselines: a fastText-style
EmbeddingBag, a TextCNN and a bidirectional GRU.  They consume raw text
columns directly so the TF-IDF vocabulary is not a hard prerequisite.  The
test split is never passed to ``fit``; vocabulary and padding are learned from
the fitting frame only.
"""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


TOKEN = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]")


class TextNeuralClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, columns, architecture='fasttext', embedding_dim=128,
                 hidden=128, epochs=30, learning_rate=0.001, weight_decay=0.01,
                 batch_size=64, max_length=256, ngram_max=2, dropout=0.2,
                 device='auto', threads=1, random_state=42, max_vocab=30000):
        self.columns = columns
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_length = max_length
        self.ngram_max = ngram_max
        self.dropout = dropout
        self.device = device
        self.threads = threads
        self.random_state = random_state
        self.max_vocab = max_vocab

    def __setstate__(self, state):
        self.__dict__.update(type(self)(columns=[]).get_params(deep=False))
        self.__dict__.update(state)

    def _device(self):
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly requested but unavailable')
        return 'cuda' if self.device != 'cpu' and torch.cuda.is_available() else 'cpu'

    def _documents(self, X):
        frame = X[self.columns].fillna('').astype(str)
        return frame.agg(' '.join, axis=1).tolist()

    @staticmethod
    def _words(document):
        return TOKEN.findall(str(document).lower())

    def _tokens(self, document):
        words = self._words(document)
        if self.architecture != 'fasttext' or self.ngram_max < 2:
            return words or ['<unk>']
        values = list(words)
        for size in range(2, self.ngram_max + 1):
            values.extend(' '.join(words[i:i + size]) for i in range(len(words) - size + 1))
        return values or ['<unk>']

    def _build_vocab(self, documents):
        from collections import Counter
        counts = Counter()
        for document in documents:
            counts.update(self._tokens(document))
        limit = max(2, self.max_vocab - 2)
        vocab = {'<pad>': 0, '<unk>': 1}
        for token, _ in counts.most_common(limit):
            vocab[token] = len(vocab)
        self.vocabulary_ = vocab

    def _encode(self, documents):
        sequences = []
        for document in documents:
            ids = [self.vocabulary_.get(token, 1) for token in self._tokens(document)]
            sequences.append(ids[:self.max_length] or [1])
        return sequences

    def _network(self):
        import torch
        from torch import nn

        if self.architecture == 'fasttext':
            class FastText(nn.Module):
                def __init__(self, vocabulary, embedding_dim, hidden, classes, dropout):
                    super().__init__()
                    self.bag = nn.EmbeddingBag(vocabulary, embedding_dim, mode='mean', padding_idx=0)
                    self.layers = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, hidden),
                                                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

                def forward(self, flat, offsets):
                    return self.layers(self.bag(flat, offsets))
            return FastText(len(self.vocabulary_), self.embedding_dim, self.hidden,
                            len(self.classes_), self.dropout)

        if self.architecture == 'textcnn':
            class TextCNN(nn.Module):
                def __init__(self, vocabulary, embedding_dim, hidden, classes, dropout, max_length):
                    super().__init__()
                    self.embedding = nn.Embedding(vocabulary, embedding_dim, padding_idx=0)
                    self.convolutions = nn.ModuleList([
                        nn.Conv1d(embedding_dim, hidden, kernel_size=size, padding=size // 2)
                        for size in (3, 4, 5)
                    ])
                    self.dropout = nn.Dropout(dropout)
                    self.classifier = nn.Linear(hidden * 3, classes)

                def forward(self, tokens):
                    values = self.embedding(tokens).transpose(1, 2)
                    pooled = [torch.max(torch.relu(conv(values)), dim=2).values
                              for conv in self.convolutions]
                    return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
            return TextCNN(len(self.vocabulary_), self.embedding_dim, self.hidden,
                           len(self.classes_), self.dropout, self.max_length)

        if self.architecture == 'textrnn':
            class TextRNN(nn.Module):
                def __init__(self, vocabulary, embedding_dim, hidden, classes, dropout):
                    super().__init__()
                    self.embedding = nn.Embedding(vocabulary, embedding_dim, padding_idx=0)
                    self.rnn = nn.GRU(embedding_dim, hidden, batch_first=True, bidirectional=True)
                    self.dropout = nn.Dropout(dropout)
                    self.classifier = nn.Linear(hidden * 4, classes)

                def forward(self, tokens):
                    values = self.embedding(tokens)
                    values, _ = self.rnn(values)
                    mask = (tokens != 0).unsqueeze(-1)
                    maximum = values.masked_fill(~mask, -1e9).max(dim=1).values
                    mean = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                    return self.classifier(self.dropout(torch.cat([maximum, mean], dim=1)))
            return TextRNN(len(self.vocabulary_), self.embedding_dim, self.hidden,
                           len(self.classes_), self.dropout)

        raise ValueError(f'Unknown text architecture: {self.architecture}')

    def _batch(self, sequences, device):
        import torch
        if self.architecture == 'fasttext':
            flat, offsets = [], []
            for sequence in sequences:
                offsets.append(len(flat))
                flat.extend(sequence)
            return (torch.tensor(flat, dtype=torch.long, device=device),
                    torch.tensor(offsets, dtype=torch.long, device=device))
        width = max(len(sequence) for sequence in sequences)
        padded = [sequence + [0] * (width - len(sequence)) for sequence in sequences]
        return (torch.tensor(padded, dtype=torch.long, device=device),)

    def fit(self, X, y):
        import torch
        from torch import nn

        device = self._device()
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.classes_, encoded = np.unique(np.asarray(y, dtype=str), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Text neural models require at least two classes')
        self._build_vocab(self._documents(X))
        sequences = self._encode(self._documents(X))
        self.n_features_in_ = len(self.columns)
        network = self._network().to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        generator = np.random.default_rng(self.random_state)
        targets = torch.tensor(encoded, dtype=torch.long, device=device)
        losses = []
        for _ in range(max(1, int(self.epochs))):
            network.train()
            order = generator.permutation(len(sequences))
            total = torch.zeros((), device=device)
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = self._batch([sequences[index] for index in indices], device)
                target = targets[torch.tensor(indices, device=device)]
                optimizer.zero_grad(set_to_none=True)
                logits = network(*batch)
                loss = criterion(logits, target)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                total += loss.detach().float() * len(indices)
            value = float(total / len(sequences))
            if not np.isfinite(value):
                raise ValueError('Text neural training loss became nonfinite')
            losses.append(value)
        self.weights_ = {key: value.detach().cpu().numpy().copy()
                         for key, value in network.state_dict().items()}
        self.training_report_ = {
            'device': device, 'architecture': self.architecture, 'epochs': len(losses),
            'training_loss': losses, 'parameter_count': sum(p.numel() for p in network.parameters()),
            'vocabulary_size': len(self.vocabulary_), 'gpu_name': torch.cuda.get_device_name()
            if device == 'cuda' else None,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
        }
        return self

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, ['weights_', 'classes_', 'vocabulary_'])
        device = self._device()
        network = self._network().to(device)
        network.load_state_dict({key: torch.from_numpy(value).to(device)
                                 for key, value in self.weights_.items()})
        network.eval()
        sequences = self._encode(self._documents(X))
        batches = []
        with torch.inference_mode():
            for start in range(0, len(sequences), self.batch_size):
                batch = self._batch(sequences[start:start + self.batch_size], device)
                batches.append(torch.softmax(network(*batch).float(), dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
