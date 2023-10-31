from mate import distribution


def test_beta_basic():
    alpha = 0.5
    beta = 2.0
    dist = distribution.Beta(alpha=alpha, beta=beta)
    assert dist.mean == alpha / (alpha + beta)
    assert dist.var == alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
