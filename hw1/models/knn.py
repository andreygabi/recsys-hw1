"""Item cosine similarity with co-occurrence shrinkage."""

import numpy as np
from scipy import sparse

from data import interaction_matrix


class ItemKNN:
    def __init__(self, neighbors=100, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, frame, shape, **kwargs):
        self.history = interaction_matrix(frame, shape, positive_only=True)
        cooccurrence = (self.history.T @ self.history).tocoo()
        norms = np.sqrt(np.asarray(self.history.sum(axis=0)).ravel())
        keep = cooccurrence.row != cooccurrence.col
        rows, cols, counts = cooccurrence.row[keep], cooccurrence.col[keep], cooccurrence.data[keep]
        similarities = counts / np.maximum(norms[rows] * norms[cols], 1e-12)
        similarities *= counts / (counts + self.shrinkage)
        full = sparse.csr_matrix((similarities, (rows, cols)), shape=(shape[1], shape[1]))
        out_rows, out_cols, out_values = [], [], []
        for item in range(shape[1]):
            start, end = full.indptr[item:item + 2]
            ids, values = full.indices[start:end], full.data[start:end]
            best = np.lexsort((ids, -values))[:self.neighbors]
            out_rows.extend([item] * len(best))
            out_cols.extend(ids[best])
            out_values.extend(values[best])
        self.similarity = sparse.csr_matrix((out_values, (out_rows, out_cols)), shape=full.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity).toarray()
