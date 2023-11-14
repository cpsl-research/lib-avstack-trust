from typing import Union

import numpy as np
from avstack.config import ALGORITHMS, MODELS
from avstack.datastructs import PriorityQueue

from mate.distribution import _Distribution


EPS = 1e-6


class _TrustEstimator:
    def propagate(self, timestamp):
        raise NotImplementedError

    def update(self, timestamp, trust):
        raise NotImplementedError


@ALGORITHMS.register_module()
class MaximumLikelihoodTrustEstimator(_TrustEstimator):
    def __init__(
        self,
        distribution: _Distribution,
        update_rate: float = 10,
        time_window: Union[int, None] = None,
        forgetting: float = 0.0,
        max_variance: float = None,
        n_min_update: int = 5,
        update_weighting: str = "uniform",
    ) -> None:
        """Estimate distribution based on maximum likelihood over a window

        inputs:
            dist: the distribution to model the trust on
            update_rate: the rate at which we fit the model to the available data
            time_window: the amount of time back to use trust measurements
            forgetting: the amount of variance inflation to add during propagation (amt / second)
        """
        self.dist = MODELS.build(distribution)
        self.update_rate = update_rate
        self.t_last_update = None
        self.t_last_prop = None
        self.time_window = time_window
        self.forgetting = forgetting
        self.variance_scaling = 1.0
        self.max_variance = max_variance
        self.n_min_update = n_min_update
        self.update_weighting = update_weighting
        self.buffer = PriorityQueue(max_size=None, max_heap=False)

    def propagate(self, timestamp):
        """Propagation model for the trust distribution
        
        TODO: propagation should interpolate back to uniform distribution
        TODO: figure out the phi, var of the uniform and interpolate (?)
        TODO: max variance should be that which comes of the uniform (?)
        """
        if self.forgetting != 0:
            if self.t_last_prop is None:
                self.t_last_prop = timestamp
            dt = timestamp - self.t_last_prop
            if dt < 0:
                raise RuntimeError("dt must be greater than or equal 0")
            elif dt > 0:
                self.variance_scaling = 1.0 + self.forgetting * dt
                if self.max_variance:
                    var = min(self.max_variance, self.dist.var * self.variance_scaling)
                else:
                    var = self.dist.var * self.variance_scaling
                self.dist.set_via_moments(self.dist.mean, var)
        self.t_last_prop = timestamp

    def update(self, timestamp, trust):
        """Update model for the trust distribution"""
        self.propagate(timestamp)
        if trust:
            self.buffer.push(priority=timestamp, item=trust)
        if self.time_window:
            self.buffer.pop_all_below(priority_max=timestamp - self.time_window)
        if not self.t_last_update:
            self.t_last_update = timestamp
        if (timestamp - self.t_last_update + EPS) >= 1.0 / self.update_rate:
            if len(self.buffer) >= self.n_min_update:
                all_x = np.array([v[1] for v in self.buffer.heap])
                if self.update_weighting == "uniform":
                    pass
                else:
                    raise NotImplementedError(self.update_weighting)
                self.dist = self.dist.fit(all_x)
                if self.max_variance:
                    if self.dist.var > self.max_variance:
                        self.dist.set_via_moments(self.dist.mean, self.max_variance)
                self.t_last_update = timestamp
            else:
                pass


@ALGORITHMS.register_module()
class ParticleFilterTrustEstimator(_TrustEstimator):
    pass


@ALGORITHMS.register_module()
class GaussianMixtureTrustEstimator(_TrustEstimator):
    pass
