"""A shared full-catalog top-K evaluation protocol."""

import numpy as np

from data import interaction_matrix


def evaluation_users(history, heldout, shape):
    positives = interaction_matrix(history, shape, positive_only=True)
    truth = interaction_matrix(heldout, shape, positive_only=True)
    return np.flatnonzero((np.diff(positives.indptr) >= 2) & (np.diff(truth.indptr) > 0))


def recommend(model, users, known, k=10, batch_size=128):
    result = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(model.score(batch), dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite model scores')
        mask = known[batch].tocoo()
        scores[mask.row, mask.col] = -np.inf
        # Stable sorting breaks score ties by the fixed item ID order.
        top = np.argsort(-scores, axis=1, kind='stable')[:, :k]
        if not np.isfinite(np.take_along_axis(scores, top, axis=1)).all():
            raise ValueError('Fewer than K unseen candidates')
        result.append(top)
    return np.concatenate(result)


def metric_values(recommendations, users, truth, k=10):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ndcg, recall = [], []
    for user, top in zip(users, recommendations):
        relevant = truth.indices[truth.indptr[user]:truth.indptr[user + 1]]
        if not len(relevant):
            raise ValueError('Evaluation user has no relevant items')
        hits = np.isin(top, relevant)
        ndcg.append(float(hits @ discounts / discounts[:min(k, len(relevant))].sum()))
        recall.append(float(hits.sum() / len(relevant)))
    return np.asarray(ndcg), np.asarray(recall)


def evaluate(model, users, history, heldout, shape):
    known = interaction_matrix(history, shape)
    truth = interaction_matrix(heldout, shape, positive_only=True)
    recommendations = recommend(model, users, known)
    ndcg, recall = metric_values(recommendations, users, truth)
    return {'NDCG@10': float(ndcg.mean()), 'Recall@10': float(recall.mean()),
            'Coverage@10': float(len(np.unique(recommendations)) / shape[1])}, recommendations, ndcg
