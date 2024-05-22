from typing import Union

import numpy as np
from avstack.config import ConfigDict

from mate.config import MATE
from mate.distributions import TrustBetaDistribution, TrustDistribution


class DistributionPropagator:
    def propagate(self, dist: TrustDistribution):
        raise NotImplementedError


@MATE.register_module()
class UncertaintyPropagator(DistributionPropagator):
    def __init__(self, delta_v: float = 0.05, v_min: float = 0, v_max: float = np.inf):
        self.delta_v = delta_v
        self.v_min = v_min
        self.v_max = v_max

    def propagate(self, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            m = dist.mean
            v = min(self.v_max, max(self.v_min, dist.precision + self.delta_v))
            dist.a = m * v
            dist.b = (1 - m) * v
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class MeanPropagator(DistributionPropagator):
    def __init__(self, delta_m: float = 30, m_min: float = 0, m_max: float = 1.0):
        self.delta_m = delta_m
        self.m_min = m_min
        self.m_max = m_max

    def propagate(self, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            m = min(
                self.m_max,
                max(self.m_min, dist.mean + (0.5 - dist.mean) / self.delta_m),
            )
            v = dist.precision
            dist.a = m * v
            dist.b = (1 - m) * v
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class PriorInterpolationPropagator(DistributionPropagator):
    def __init__(
        self, prior: Union[ConfigDict, TrustDistribution], w_prior: float = 0.2
    ):
        self.w_prior = w_prior
        self.prior = (
            prior if isinstance(prior, TrustDistribution) else MATE.build(prior)
        )

    def propagate(self, dist: TrustDistribution):
        w = self.w_prior
        if isinstance(dist, TrustBetaDistribution):
            dist.alpha = (1 - w) * dist.alpha + w * self.prior.alpha
            dist.beta = (1 - w) * dist.beta + w * self.prior.beta
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class NormalizationPropagator(DistributionPropagator):
    def __init__(self, s_factor: float = 2.0):
        self.s_factor = s_factor

    def propagate(self, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            s = dist.precision / self.s_factor
            dist.alpha /= s
            dist.beta /= s
        else:
            raise NotImplementedError(type(dist))
