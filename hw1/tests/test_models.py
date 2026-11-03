import unittest

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from data import interaction_matrix
from evaluation import metric_values, recommend
from models import CustomTargetALS, EASE, FM, IALS, ItemKNN
from models.ials import objective, update_factors


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.frame = pd.DataFrame({'u': [0, 0, 1, 1, 2, 2], 'i': [0, 1, 1, 2, 0, 2],
                                   'rating': [5., 2., 4., 5., 3., 5.],
                                   'timestamp': [100, 110, 120, 130, 140, 150]})
        self.shape = (3, 4)
        self.codes = {'brand': np.array([0, 0, 1, 1]), 'category': np.array([0, 1, 0, 1])}

    def test_als_against_dense_normal_equations(self):
        target = interaction_matrix(self.frame, self.shape, values=(self.frame.rating >= 4).astype(float))
        fixed = np.random.default_rng(1).normal(size=(4, 3))
        fitted = update_factors(target, fixed, 4, 0.7)
        confidence = np.ones(self.shape)
        confidence[self.frame.u, self.frame.i] += 4
        for user in range(3):
            matrix = fixed.T @ np.diag(confidence[user]) @ fixed + 0.7 * np.eye(3)
            rhs = fixed.T @ (confidence[user] * target.toarray()[user])
            np.testing.assert_allclose(fitted[user], np.linalg.solve(matrix, rhs))
        dense_loss = np.sum(confidence * (target.toarray() - fitted @ fixed.T)**2)
        dense_loss += 0.7 * (np.sum(fitted**2) + np.sum(fixed**2))
        self.assertAlmostEqual(dense_loss, objective(target, fitted, fixed, 4, 0.7))

    def test_als_loss_decreases(self):
        for model in (IALS(factors=3, iterations=5), CustomTargetALS(factors=3, iterations=5)):
            model.fit(self.frame, self.shape, cutoff=151)
            self.assertTrue(np.all(np.diff(model.losses) <= 1e-8))

    def test_custom_target_is_signed_and_uses_only_history(self):
        model = CustomTargetALS()
        target = model.training_target(self.frame, self.shape, 151)
        self.assertLess(target[0, 1], 0)
        self.assertGreater(target[0, 0], 0)
        self.assertEqual(target[0, 3], 0)
        self.assertEqual(target.nnz, len(self.frame))
        later = model.training_target(self.frame, self.shape, 151 + 86400 * 730)
        self.assertLess(abs(later[0, 0]), abs(target[0, 0]))
        with self.assertRaises(ValueError):
            model.training_target(self.frame, self.shape, 90)

    def test_ease_stationary_point_and_diagonal(self):
        model = EASE(regularization=2).fit(self.frame, self.shape)
        x = interaction_matrix(self.frame, self.shape, positive_only=True).toarray()
        gradient = (x.T @ x + 2 * np.eye(4)) @ model.weights - x.T @ x
        np.fill_diagonal(gradient, 0)
        np.testing.assert_allclose(gradient, 0, atol=1e-6)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_knn_cosine_and_shrinkage(self):
        model = ItemKNN(neighbors=3, shrinkage=2).fit(self.frame, self.shape)
        self.assertAlmostEqual(model.similarity[1, 2], (1 / np.sqrt(2)) / 3)
        np.testing.assert_array_equal(model.similarity.diagonal(), 0)

    def test_fm_fast_scores_equal_all_pair_interactions(self):
        for metadata in (False, True):
            model = FM(factors=3, epochs=1, negative_ratio=1, use_metadata=metadata)
            model.fit(self.frame, self.shape, self.codes)
            users = np.repeat(np.arange(3), 4)
            items = np.tile(np.arange(4), 3)
            features = torch.as_tensor(model.features(users, items))
            with torch.no_grad():
                vectors = model.embedding(features)
                brute = model.bias + model.linear(features).sum((1, 2))
                for a in range(features.shape[1]):
                    for b in range(a + 1, features.shape[1]):
                        brute += (vectors[:, a] * vectors[:, b]).sum(1)
            np.testing.assert_allclose(model.score(np.arange(3)).ravel(), brute.numpy(), atol=1e-6)

    def test_ranking_masks_all_ratings_and_metrics(self):
        class ConstantModel:
            def score(self, users):
                return np.ones((len(users), 4))
        known = interaction_matrix(self.frame, self.shape)
        top = recommend(ConstantModel(), np.array([0]), known, k=2)
        np.testing.assert_array_equal(top, [[2, 3]])
        truth = sparse.csr_matrix(([1, 1], ([0, 0], [1, 2])), shape=self.shape)
        ndcg, recall = metric_values(np.array([[2, 3]]), [0], truth, k=2)
        self.assertAlmostEqual(ndcg[0], 1 / (1 + 1 / np.log2(3)))
        self.assertEqual(recall[0], 0.5)


if __name__ == '__main__':
    unittest.main()
