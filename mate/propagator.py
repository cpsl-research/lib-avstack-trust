from typing import Union

import numpy as np
from avstack.config import ConfigDict

from mate.config import MATE
from mate.distributions import TrustBetaDistribution, TrustDistribution


class DistributionPropagator:
    def propagate(self, timestamp: float, dist: TrustDistribution):
        dt = timestamp - dist.timestamp
        if dt > 0:
            self._propagate(dt, dist)
        dist.timestamp = timestamp

    def _propagate(self, dt: float, dist: TrustDistribution):
        raise NotImplementedError


@MATE.register_module()
class UncertaintyPropagator(DistributionPropagator):
    def __init__(self, delta_v: float = 0.05, v_min: float = 0, v_max: float = np.inf):
        """Propagate the uncertainty of a trust distribution

        Equation:
        v = v + delta_v * dt

        Args:
            delta_v : increase in variance (units of 1/s)
            v_min : minimum variance to allow
            v_max : maximum variance to allow
        """
        self.delta_v = delta_v
        self.v_min = v_min
        self.v_max = v_max

    def _propagate(self, dt: float, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            m = dist.mean
            v = min(self.v_max, max(self.v_min, dist.precision + self.delta_v * dt))
            dist.a = m * v
            dist.b = (1 - m) * v
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class MeanPropagator(DistributionPropagator):
    def __init__(self, factor_m: float = 30, m_min: float = 0, m_max: float = 1.0):
        """Propagate the mean of a trust distribution towards 0.5

        Equation:
        m = m + (.5 - m) / (factor_m*dt)

        Args:
            factor_m : factor to move towards 1/2 (larger = slower) (units of 1/s)
            m_min : minimum mean to allow
            m_max : maximum mean to allow
        """
        self.factor_m = factor_m
        self.m_min = m_min
        self.m_max = m_max

    def _propagate(self, dt: float, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            m = min(
                self.m_max,
                max(self.m_min, dist.mean + (0.5 - dist.mean) / (self.factor_m * dt)),
            )
            v = dist.precision
            dist.a = m * v
            dist.b = (1 - m) * v
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class PriorInterpolationPropagator(DistributionPropagator):
    def __init__(
        self, prior: Union[ConfigDict, TrustDistribution], dt_return: float = 10.0
    ):
        """Propagate the distribution towards a prior distribution

        Equation
        dist = dt/dt_return * dist + (1-dt/dt_return) * prior

        Args:
            prior : the prior distribution to propagate towards
            dt_return : the amount of time without update to return to the prior completely
        """
        self.dt_return = dt_return
        self.prior = (
            prior if isinstance(prior, TrustDistribution) else MATE.build(prior)
        )

    def _propagate(self, dt: float, dist: TrustDistribution):
        w = min(1.0, max(0.0, dt / self.dt_return))
        if isinstance(dist, TrustBetaDistribution):
            dist.alpha = (1 - w) * dist.alpha + w * self.prior.alpha
            dist.beta = (1 - w) * dist.beta + w * self.prior.beta
        else:
            raise NotImplementedError(type(dist))


@MATE.register_module()
class NormalizationPropagator(DistributionPropagator):
    def __init__(self, parameter_sum: float = 10.0, mean_eps: float = 0.05):
        """Normalize the parameters of a distribution; time-independent

        Keeps the mean constant and find the parameters that minimize
        the difference in variance between original and new such that
        the sum of parameters is preserved.

        Minimization Equation:
          Beta:
            min (v_orig - v_new(a,b))^2
            where v_new(a,b) = a*b / ((a+b)^2 * (a+b+1))
            s.t.  |a / (a + b) - m_orig| < mean_eps
                  a + b = parameter_sum
                  a > 0, b > 0

        Args:
            parameter_sum : the desired sum of parameters
        """
        self.parameter_sum = parameter_sum

    def _propagate(self, dt: float, dist: TrustDistribution):
        if isinstance(dist, TrustBetaDistribution):
            raise NotImplementedError
        else:
            raise NotImplementedError(type(dist))
