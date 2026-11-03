"""Alternating least squares with a dense unobserved-zero term."""

import numpy as np
from scipy.linalg import solve

from data import interaction_matrix


def update_factors(target, fixed, alpha, regularization):
    gram = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
    result = np.empty((target.shape[0], fixed.shape[1]))
    for row in range(target.shape[0]):
        start, end = target.indptr[row:row + 2]
        indices, values = target.indices[start:end], target.data[start:end]
        observed = fixed[indices]
        matrix = gram + alpha * (observed.T @ observed)
        rhs = (1 + alpha) * (observed.T @ values)
        result[row] = solve(matrix, rhs, assume_a='pos', check_finite=False)
    return result


def objective(target, users, items, alpha, regularization):
    coordinates = target.tocoo()
    predictions = np.einsum('ij,ij->i', users[coordinates.row], items[coordinates.col])
    dense_zero_loss = np.sum((users.T @ users) * (items.T @ items))
    correction = np.sum((1 + alpha) * (coordinates.data - predictions)**2 - predictions**2)
    return float(dense_zero_loss + correction + regularization * (np.sum(users**2) + np.sum(items**2)))


class IALS:
    def __init__(self, factors=32, alpha=10, regularization=1, iterations=10, seed=42):
        self.factors, self.alpha = factors, alpha
        self.regularization, self.iterations, self.seed = regularization, iterations, seed

    def training_target(self, frame, shape, cutoff=None):
        values = (frame.rating.to_numpy() >= 4).astype(float)
        return interaction_matrix(frame, shape, values=values)

    def fit(self, frame, shape, cutoff=None, **kwargs):
        target = self.training_target(frame, shape, cutoff)
        transposed = target.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.users = rng.normal(0, 0.05, (shape[0], self.factors))
        self.items = rng.normal(0, 0.05, (shape[1], self.factors))
        self.losses = [objective(target, self.users, self.items, self.alpha, self.regularization)]
        for _ in range(self.iterations):
            self.users = update_factors(target, self.items, self.alpha, self.regularization)
            self.items = update_factors(transposed, self.users, self.alpha, self.regularization)
            self.losses.append(objective(target, self.users, self.items, self.alpha, self.regularization))
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
