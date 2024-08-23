import numpy as np

from avtrust.distributions import TrustBetaDistribution
from avtrust.propagator import (
    MeanPropagator,
    PrecisionPropagator,
    PriorInterpolationPropagator,
    VariancePropagator,
)


def test_mean_propagator():
    dist = TrustBetaDistribution(timestamp=0, identifier=0, alpha=5, beta=1)
    m1, p1, v1 = dist.mean, dist.precision, dist.variance
    prop = MeanPropagator(factor_m=30, m_min=0, m_max=1)
    prop.propagate(timestamp=1.0, dist=dist)
    m2, p2, v2 = dist.mean, dist.precision, dist.variance
    assert dist.timestamp == 1.0
    assert abs(m1 - 0.5) > abs(m2 - 0.5)
    assert np.isclose(p1, p2)


def test_precision_propagator():
    dist = TrustBetaDistribution(timestamp=0, identifier=0, alpha=5, beta=1)
    m1, p1, v1 = dist.mean, dist.precision, dist.variance
    prop = PrecisionPropagator(delta_p=0.5, p_min=0)
    prop.propagate(timestamp=1.0, dist=dist)
    m2, p2, v2 = dist.mean, dist.precision, dist.variance
    assert dist.timestamp == 1.0
    assert p2 > p1
    assert np.isclose(m1, m2)


def test_variance_propagator():
    dist = TrustBetaDistribution(timestamp=0, identifier=0, alpha=5, beta=1)
    m1, p1, v1 = dist.mean, dist.precision, dist.variance
    prop = VariancePropagator(delta_v=0.1, v_min=0)
    prop.propagate(timestamp=1.0, dist=dist)
    m2, p2, v2 = dist.mean, dist.precision, dist.variance
    assert dist.timestamp == 1.0
    assert v2 > v1
    assert np.isclose(m1, m2)


def test_prior_propagator():
    prior = TrustBetaDistribution(timestamp=0, identifier=0, alpha=1, beta=1)
    dist = TrustBetaDistribution(timestamp=0, identifier=0, alpha=5, beta=1)
    m1, p1, v1 = dist.mean, dist.precision, dist.variance
    prop = PriorInterpolationPropagator(prior=prior, dt_return=10)
    prop.propagate(timestamp=1.0, dist=dist)
    m2, p2, v2 = dist.mean, dist.precision, dist.variance
    assert dist.timestamp == 1.0
    assert abs(m1 - prior.mean) > abs(m2 - prior.mean)
    assert abs(v1 - prior.variance) > abs(v2 - prior.variance)
