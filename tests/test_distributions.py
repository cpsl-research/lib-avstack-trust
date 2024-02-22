import numpy as np

from mate import distribution


def test_beta_basic():
    alpha = 0.5
    beta = 2.0
    dist = distribution.Beta(alpha=alpha, beta=beta)
    assert dist.mean == alpha / (alpha + beta)
    assert dist.var == alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))


def test_beta_sample():
    mean = 0.5
    var = 0.1**2
    dist = distribution.Beta(mean=mean, var=var)
    samples = dist.rvs(n=10000)
    assert abs(np.mean(samples) - mean) < 0.01
    assert abs(np.std(samples) - np.sqrt(var)) < 0.01


def test_beta_fit():
    alpha = 1.0
    beta = 2.0
    dist1 = distribution.Beta(alpha=alpha, beta=beta)
    samples = dist1.rvs(n=100000, random_state=123)
    dist2 = distribution.Beta.fit(samples)
    assert abs(alpha - dist2.alpha) < 0.1 and abs(beta - dist2.beta) < 0.1


def test_uniform_basic():
    a = 5.0
    b = 6.5
    dist = distribution.Uniform(a=a, b=b)
    assert np.isclose(dist.mean, (a + b) / 2)
    assert dist.pdf(a - 1.0) == 0.0
    assert dist.cdf(a - 1.0) == 0.0
    assert dist.cdf(b + 1.0) == 1.0
    assert dist.pdf(a), 1 / (b - a)
