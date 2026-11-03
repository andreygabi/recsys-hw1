"""Closed-form linear reconstruction with a zero diagonal."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from data import interaction_matrix


class EASE:
    def __init__(self, regularization=200):
        self.regularization = regularization

    def fit(self, frame, shape, **kwargs):
        self.history = interaction_matrix(frame, shape, positive_only=True)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        inverse = cho_solve(cho_factor(gram, check_finite=False), np.eye(shape[1]), check_finite=False)
        weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(weights, 0)
        self.weights = weights.astype(np.float32)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
