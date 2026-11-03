"""ALS with signed, user-centered and time-decayed preferences."""

import numpy as np

from data import interaction_matrix
from .ials import IALS


class CustomTargetALS(IALS):
    def __init__(self, beta=1, half_life_days=730, **kwargs):
        super().__init__(**kwargs)
        self.beta, self.half_life_days = beta, half_life_days

    def training_target(self, frame, shape, cutoff=None):
        if cutoff is None or cutoff < frame.timestamp.max():
            raise ValueError('cutoff must be at or after every training event')
        ratings = frame.rating.to_numpy()
        users = frame.u.to_numpy()
        counts = np.bincount(users, minlength=shape[0])
        sums = np.bincount(users, weights=ratings, minlength=shape[0])
        means = (sums + 5 * ratings.mean()) / (counts + 5)
        preference = np.tanh((ratings - 3) / 2 + self.beta * (ratings - means[users]) / 2)
        age = (cutoff - frame.timestamp.to_numpy()) / 86400
        decay = 0.5 + 0.5 * np.exp2(-age / self.half_life_days)
        return interaction_matrix(frame, shape, values=preference * decay)
