import numpy as np

from mate import distribution


def test_beta_basic():
    alpha = 0.5
    beta = 2.0
    dist = distribution.Beta(alpha=alpha, beta=beta)
    assert dist.mean == alpha / (alpha + beta)
    assert dist.var == alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))


def test_beta_fit():
    alpha = 1.0
    beta = 2.0
    dist1 = distribution.Beta(alpha=alpha, beta=beta)
    samples = dist1.rvs(n=100000, random_state=123)
    dist2 = distribution.Beta.fit(samples)
    assert abs(alpha - dist2.alpha)<0.1 and abs(beta - dist2.beta)<0.1