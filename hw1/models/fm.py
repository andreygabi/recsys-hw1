"""Second-order factorization machine, trained with sampled binary loss."""

import numpy as np
import torch
from torch import nn


class FM:
    def __init__(self, factors=32, regularization=1e-5, epochs=12, learning_rate=0.01,
                 negative_ratio=3, batch_size=4096, seed=42, use_metadata=True):
        self.factors, self.regularization, self.epochs = factors, regularization, epochs
        self.learning_rate, self.negative_ratio = learning_rate, negative_ratio
        self.batch_size, self.seed, self.use_metadata = batch_size, seed, use_metadata

    def features(self, users, items):
        columns = [np.asarray(users), self.shape[0] + np.asarray(items)]
        if self.use_metadata:
            offset = sum(self.shape)
            for field in ('brand', 'category'):
                columns.append(offset + self.codes[field][items])
                offset += int(self.codes[field].max()) + 1
        return np.column_stack(columns)

    def logits(self, features):
        vectors = self.embedding(features)
        interactions = 0.5 * (vectors.sum(1).square() - vectors.square().sum(1)).sum(1)
        return self.bias + self.linear(features).sum((1, 2)) + interactions

    def fit(self, frame, shape, codes, **kwargs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.shape, self.codes = shape, codes
        num_features = sum(shape)
        if self.use_metadata:
            num_features += sum(int(codes[field].max()) + 1 for field in ('brand', 'category'))
        self.embedding = nn.Embedding(num_features, self.factors)
        self.linear = nn.Embedding(num_features, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.05)
        nn.init.zeros_(self.linear.weight)
        known = set(zip(frame.u, frame.i))
        positive_users = frame.loc[frame.rating >= 4, 'u'].to_numpy()
        negative_users = np.repeat(positive_users, self.negative_ratio)
        negative_items = rng.integers(shape[1], size=len(negative_users))
        invalid = np.fromiter(((int(u), int(i)) in known for u, i in zip(negative_users, negative_items)), bool)
        while invalid.any():
            negative_items[invalid] = rng.integers(shape[1], size=invalid.sum())
            invalid = np.fromiter(((int(u), int(i)) in known for u, i in zip(negative_users, negative_items)), bool)
        users = np.concatenate([frame.u.to_numpy(), negative_users])
        items = np.concatenate([frame.i.to_numpy(), negative_items])
        labels = np.concatenate([(frame.rating.to_numpy() >= 4).astype(np.float32),
                                 np.zeros(len(negative_users), dtype=np.float32)])
        features = torch.as_tensor(self.features(users, items), dtype=torch.long)
        labels = torch.from_numpy(labels)
        parameters = [self.bias, self.linear.weight, self.embedding.weight]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.losses = []
        for _ in range(self.epochs):
            permutation = rng.permutation(len(labels))
            total = 0.0
            for start in range(0, len(labels), self.batch_size):
                batch = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self.logits(features[batch]), labels[batch])
                penalty = self.embedding.weight.square().sum() + self.linear.weight.square().sum()
                loss = loss + self.regularization * penalty
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.losses.append(total / len(labels))
        self._prepare_scores()
        return self

    def _prepare_scores(self):
        vectors = self.embedding.weight.detach().numpy()
        linear = self.linear.weight.detach().numpy().ravel()
        self.user_vectors = vectors[:self.shape[0]]
        self.user_bias = linear[:self.shape[0]]
        item_fields = self.features(np.zeros(self.shape[1], dtype=int), np.arange(self.shape[1]))[:, 1:]
        item_vectors = vectors[item_fields]
        self.item_vectors = item_vectors.sum(1)
        pair_terms = 0.5 * (self.item_vectors**2 - (item_vectors**2).sum(1)).sum(1)
        self.item_bias = linear[item_fields].sum(1) + pair_terms + self.bias.detach().item()

    def score(self, users):
        return self.user_vectors[users] @ self.item_vectors.T + self.user_bias[users, None] + self.item_bias[None, :]
