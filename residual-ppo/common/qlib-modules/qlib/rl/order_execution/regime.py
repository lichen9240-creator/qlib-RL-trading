# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Train-only variational HMM and causal features for execution policies.

Transition rows have Dirichlet posteriors; diagonal Gaussian emissions are
point estimates (variational EM). Inference exposes filtered, never smoothed,
beliefs. Sequences are independent stock-days, so overnight transitions are
not invented. Feature normalization is shared by all ablation settings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, logsumexp
from sklearn.mixture import GaussianMixture


def causal_features(frame: pd.DataFrame) -> np.ndarray:
    """Feature at row t becomes observable only AFTER bar t has completed."""
    price = frame["$close0"].astype(float)
    volume = frame["$volume0"].astype(float)
    if not (np.isfinite(price).all() and (price > 0).all() and np.isfinite(volume).all() and (volume >= 0).all()):
        raise ValueError("Execution features require finite positive prices and nonnegative volumes")
    ret = np.log(price).diff().fillna(0)
    return np.column_stack((ret, np.log1p(volume), ret.rolling(6, min_periods=1).std(ddof=0),
                            ret.rolling(4, min_periods=1).sum()))


class BayesianRegime:
    def __init__(self, n_states=3, transition_prior=1.0, sticky_prior=2.0, variance_floor=1e-3, seed=42):
        if n_states < 2 or transition_prior <= 0 or sticky_prior < 0 or variance_floor <= 0:
            raise ValueError("Invalid HMM hyperparameters")
        self.n_states = int(n_states)
        self.prior = np.full((n_states, n_states), float(transition_prior)) + np.eye(n_states) * sticky_prior
        self.variance_floor = float(variance_floor)
        self.seed = seed
        self.metadata = {}
        self.losses = []

    def normalize(self, x):
        return (np.asarray(x) - self.center) / self.scale

    def _emissions(self, x):
        return -0.5 * (np.log(2 * np.pi * self.variance).sum(-1) +
                       (((x[..., None, :] - self.mean) ** 2) / self.variance).sum(-1))

    def _posterior(self, x, mask):
        """Batched forward/backward, used ONLY while fitting training sequences."""
        emission = self._emissions(x)
        elog_a = digamma(self.alpha) - digamma(self.alpha.sum(-1, keepdims=True))
        batch, length = mask.shape
        forward = np.empty((batch, length, self.n_states))
        forward[:, 0] = np.log(self.initial) + emission[:, 0]
        for t in range(1, length):
            forward[:, t] = emission[:, t] + logsumexp(forward[:, t - 1, :, None] + elog_a, axis=1)
        ends = mask.sum(1) - 1
        log_z = logsumexp(forward[np.arange(batch), ends], axis=-1)
        backward = np.zeros_like(forward)
        for t in range(length - 2, -1, -1):
            update = logsumexp(elog_a + emission[:, t + 1, None, :] + backward[:, t + 1, None, :], axis=-1)
            backward[:, t] = np.where(mask[:, t + 1, None], update, 0)
        gamma = np.exp(np.where(mask[..., None], forward + backward - log_z[:, None, None], -np.inf))
        counts = np.zeros_like(self.alpha)
        for t in range(length - 1):
            log_xi = (forward[:, t, :, None] + elog_a + emission[:, t + 1, None, :] +
                      backward[:, t + 1, None, :] - log_z[:, None, None])
            counts += np.exp(np.where(mask[:, t + 1, None, None], log_xi, -np.inf)).sum(0)
        kl = (gammaln(self.alpha.sum(-1)) - gammaln(self.alpha).sum(-1)
              - gammaln(self.prior.sum(-1)) + gammaln(self.prior).sum(-1)
              + ((self.alpha - self.prior) * elog_a).sum(-1)).sum()
        return gamma, counts, float(log_z.sum() - kl)

    def fit(self, sequences, max_iter=40, tol=1e-5, metadata=None):
        if max_iter < 1 or tol < 0:
            raise ValueError("max_iter must be positive and tol nonnegative")
        sequences = [np.asarray(s, dtype=np.float64) for s in sequences]
        if not sequences or any(s.ndim != 2 or s.shape[1] != 4 or not len(s) for s in sequences):
            raise ValueError("Expected nonempty stock-day sequences with four features")
        flat = np.concatenate(sequences)
        if not np.isfinite(flat).all():
            raise ValueError("Non-finite training features")
        self.center, self.scale = flat.mean(0), np.maximum(flat.std(0), 1e-6)
        x = np.zeros((len(sequences), max(map(len, sequences)), 4))
        mask = np.zeros(x.shape[:2], dtype=bool)
        for i, seq in enumerate(sequences):
            x[i, :len(seq)] = self.normalize(seq)
            mask[i, :len(seq)] = True
        gmm = GaussianMixture(self.n_states, covariance_type="diag", random_state=self.seed,
                              reg_covar=self.variance_floor).fit(self.normalize(flat))
        self.mean, self.variance = gmm.means_, np.maximum(gmm.covariances_, self.variance_floor)
        self.initial = np.full(self.n_states, 1 / self.n_states)
        self.alpha = self.prior.copy()
        self.losses = []
        gamma, counts, elbo = self._posterior(x, mask)
        self.losses.append(-elbo)
        for _ in range(max_iter):
            weights = gamma.sum((0, 1)).clip(1e-10)
            self.mean = np.einsum("btk,btd->kd", gamma, x) / weights[:, None]
            self.variance = np.maximum(
                np.einsum("btk,btkd->kd", gamma, (x[:, :, None] - self.mean) ** 2) / weights[:, None],
                self.variance_floor,
            )
            self.alpha = self.prior + counts  # do not count the same data again each iteration
            self.initial = gamma[:, 0].sum(0).clip(1e-10)
            self.initial /= self.initial.sum()
            gamma, counts, elbo = self._posterior(x, mask)
            self.losses.append(-elbo)
            if abs(self.losses[-1] - self.losses[-2]) <= tol * (1 + abs(self.losses[-2])):
                break
        self.metadata = dict(metadata or {}, sequences=len(sequences), observations=len(flat),
                             feature_version=1, seed=self.seed, variance_floor=self.variance_floor)
        return self

    def filter(self, raw_features):
        """One belief per completed bar, with no lookahead or backward pass."""
        emissions = self._emissions(self.normalize(raw_features))
        log_a = np.log(self.alpha / self.alpha.sum(-1, keepdims=True))
        previous = np.log(self.initial)
        beliefs = []
        for t, emission in enumerate(emissions):
            prediction = previous if t == 0 else logsumexp(previous[:, None] + log_a, axis=0)
            previous = prediction + emission
            previous -= logsumexp(previous)
            beliefs.append(np.exp(previous))
        return np.asarray(beliefs).reshape(-1, self.n_states)

    def save(self, path):
        np.savez_compressed(path, center=self.center, scale=self.scale, mean=self.mean, variance=self.variance,
                            initial=self.initial, alpha=self.alpha, prior=self.prior, losses=self.losses,
                            metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            model = cls(n_states=len(data["initial"]))
            for name in ("center", "scale", "mean", "variance", "initial", "alpha", "prior"):
                setattr(model, name, data[name])
            model.losses = data["losses"].tolist()
            model.metadata = json.loads(str(data["metadata"]))
        return model
