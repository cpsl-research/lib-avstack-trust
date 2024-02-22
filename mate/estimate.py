from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from mate.measurement import TrustArray, UncertainTrustArray
    from scipy.stats._result_classes import BinomTestResult

import numpy as np
from avstack.config import ALGORITHMS, MODELS, ConfigDict
from avstack.datastructs import PriorityQueue
from avstack.modules.estimation import kalman
from scipy.stats import binomtest, multivariate_normal, norm

from mate.sampling import n_eff, systematic_sampling


EPS = np.finfo(float).eps


class _TrustEstimator:
    def __init__(self, t0: float, verbose: bool = False) -> None:
        self.t0 = t0
        self.t_last_update = t0
        self.t_last_prop = t0
        self.verbose = verbose
        self.agent_ID_to_index = {}

    def mean(self, ID=None):
        raise NotImplementedError

    def variance(self, ID=None):
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    @property
    def n_agents(self):
        return len(self.agent_ID_to_index)

    def __len__(self):
        return len(self.agent_ID_to_index)

    def add_agent(
        self,
        ID: int,
        *args,
        **kwargs,
    ) -> None:
        self.agent_ID_to_index[ID] = self.n_agents
        self._add_agent(ID=ID, *args, **kwargs)

    def propagate(self, timestamp):
        dt = timestamp - self.t_last_prop
        if dt < 0:
            raise RuntimeError(
                "dt must be greater than or equal 0, got {} - {}".format(
                    timestamp, self.t_last_prop
                )
            )
        elif dt > 0:
            self._propagate(dt)
            self.t_last_prop = timestamp
        else:
            pass

    def update(self, timestamp, trust, **kwargs):
        if self.multi:
            for ID in trust.ID:
                if ID not in self.agent_ID_to_index:
                    self.add_agent(ID=ID)
        self.propagate(timestamp)
        self._update(timestamp, trust, **kwargs)

    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError("Implement in subclass")


#############################################################
# VECTOR/MATRIX IMPLEMENTATIONS
#############################################################


@ALGORITHMS.register_module()
class MultiTrustEstimatorWrapper(_TrustEstimator):
    multi = True

    def __init__(self, t0: float, estimator_base, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._estimator_base = estimator_base
        self.estimators = {}
        super().__init__(t0=t0)

    def __len__(self):
        return len(self.estimators)

    @property
    def name(self):
        return self._estimator_base.__name__

    def mean(self, ID=None):
        return self.estimators[ID].mean()

    def variance(self, ID=None):
        return self.estimators[ID].variance()

    def pdf(self, x: float, ID=None):
        return self.estimators[ID].pdf(x=x)

    def _add_agent(self, ID, *args, **kwargs) -> None:
        self.estimators[ID] = self._estimator_base(
            t0=self.t0, *args, *self._args, **kwargs, **self._kwargs
        )

    def _propagate(self, *args, **kwargs):
        for est in self.estimators.values():
            est._propagate(*args, **kwargs)

    def _update(self, timestamp, trust, *args, **kwargs):
        if len(trust) == 2:
            for tr, ID in trust:
                self.estimators[ID]._update(timestamp, trust=tr, *args, **kwargs)
        else:
            for tr, c, f, ID in trust:
                self.estimators[ID]._update(timestamp, trust=tr, confidence=c, prior=f)


class _KalmanTrustEstimator(_TrustEstimator):
    def __init__(
        self,
        t0: float,
        x0_sub: np.ndarray,
        P0_sub: np.ndarray,
        process_noise_sub: np.ndarray,
        default_confidence: 0.80,
        verbose: bool = False,
    ) -> None:
        super().__init__(t0=t0, verbose=verbose)
        self.x0_sub = np.asarray(x0_sub)
        self.P0_sub = np.asarray(P0_sub)
        assert self.P0_sub.shape == (len(self.x0_sub), len(self.x0_sub))
        self.process_noise_sub = process_noise_sub
        self.default_confidence = default_confidence
        self.x = np.zeros((0,))
        self.P = np.zeros((0, 0))
        self.n_sub_dim = len(x0_sub)

    def mean(self, ID):
        idx = self.agent_ID_to_index[ID]
        return self.x[idx * self.n_sub_dim : (idx + 1) * self.n_sub_dim]

    def variance(self, ID):
        idx = self.agent_ID_to_index[ID]
        return self.P[
            [idx * self.n_sub_dim, idx * self.n_sub_dim + 1],
            [idx * self.n_sub_dim, idx * self.n_sub_dim + 1],
        ]

    def _add_agent(self, ID: int) -> None:
        self.x = np.concatenate((self.x, self.x0_sub), axis=0)
        self.P = np.hstack((self.P, np.zeros((self.P.shape[0], self.n_sub_dim))))
        self.P = np.vstack((self.P, np.zeros((self.n_sub_dim, self.P.shape[1]))))
        self.P[-self.n_sub_dim : 0, -self.n_sub_dim : 0] = self.P0_sub

    def _propagate(self, dt: float) -> None:
        F_func = lambda dt: np.eye(self.n_sub_dim * self.n_agents)
        Q_func = lambda dt: dt * np.kron(self.process_noise_sub, np.eye(self.n_agents))
        self.x, self.P = kalman.kalman_linear_predict(
            self.x, self.P, F_func, Q_func, dt
        )


@ALGORITHMS.register_module()
class DirectLinearKalmanTrustEstimator(_KalmanTrustEstimator):
    """Approximates trust as Gaussian even though it is on [0, 1]"""

    multi = True

    def __init__(
        self,
        t0: float,
        t_prior: float = 0.70,
        p_prior: float = 0.20,
        process_noise: float = 0.20,
        default_confidence: float = 0.70,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            t0=t0,
            x0_sub=np.array([t_prior]),
            P0_sub=np.array([[p_prior]]),
            process_noise_sub=np.array([[process_noise]]),
            default_confidence=default_confidence,
            verbose=verbose,
        )

    def pdf(self, x: Union[float, np.ndarray], ID=None):
        if ID is None:
            # pdf of joint
            pdf = multivariate_normal.pdf(x=x, mean=self.x, cov=self.P)
        else:
            # pdf of marginal
            idx = self.agent_ID_to_index[ID]
            pdf = norm.pdf(x=x, loc=self.x[idx], scale=self.P[idx, idx])
        return pdf

    def _update(
        self,
        timestamp: float,
        trust: Union["UncertainTrustArray", "TrustArray"],
    ) -> None:
        z = trust.trust
        try:
            R = np.diag(1 - trust.confidence)
        except AttributeError:
            R = np.diag(1 - self.default_confidence * np.ones((len(z),)))
        H = np.zeros((self.n_agents, self.n_agents))  # mapping of state to measurement
        for i, ID in enumerate(trust.ID):
            H[i, self.agent_ID_to_index[ID]] = 1.0
        self.x, self.P = kalman.kalman_linear_update(self.x, self.P, z, H, R)


@ALGORITHMS.register_module()
class HierarchicalExtendedKalmanTrustEstimator(_TrustEstimator):
    """Models the parameters of the Beta with Gaussians"""

    multi = True


@ALGORITHMS.register_module()
class HierarchicalUnscentedKalmanTrustEstimator(_TrustEstimator):
    """Models the parameters of the Beta with Gaussians"""

    multi = True

    def __init__(
        self,
        t0: float,
        x0: np.ndarray = np.array([5, 5]),
        p0: np.ndarray = np.diag([0.5, 0.5]),
        process_noise: np.ndarray = np.diag([0.5, 0.5]),
        default_confidence: float = 0.70,
        verbose: bool = False,
    ) -> None:
        """Parameters are phi and lambda

        phi    -> alpha / (alpha + beta)
        lambda -> alpha + beta
        """
        super().__init__(
            t0=t0,
            x0_sub=x0,
            P0_sub=p0,
            process_noise_sub=process_noise,
            default_confidence=default_confidence,
            verbose=verbose,
        )

    def _update(
        self,
        timestamp: float,
        trust: Union["UncertainTrustArray", "TrustArray"],
    ) -> None:
        """We assume z is a measurement of the mean of the Beta distribution
        and the confidence is a measurement 
         
        We must create the
        measurement function that maps the state to the measurement space

        Mean of Beta is alpha / (alpha + beta)
        In (phi, lam) parameterization, mean is straight phi
        """
        z = trust.trust
        h_func = lambda x: None
        try:
            R = np.diag(1 - trust.confidence)
        except AttributeError:
            R = np.diag(1 - self.default_confidence * np.ones((len(z),)))
        

        # def kalman_unscented_update(xp, Pp, z, h_func, R):
        # H function is a


@ALGORITHMS.register_module()
class GaussianCopulaTrustEstimator(_TrustEstimator):
    """A Gaussian Copula is used to model the correlations
    between the variables while ensuring that the marginals
    remain Beta distributed.

    Original inspiration from:
    https://stats.stackexchange.com/questions/87358/how-to-construct-a-multivariate-beta-distribution
    https://twiecki.io/blog/2018/05/03/copulas/
    """

    multi = True

    def __init__(
        self,
        t0: float,
        x0: np.ndarray = np.array([5, 5]),
        p0: np.ndarray = np.diag([0.5, 0.5]),
        process_noise: np.ndarray = np.diag([0.5, 0.5]),
        default_confidence: float = 0.70,
        verbose: bool = False,
    ) -> None:
        """Parameters are phi and lambda

        phi    -> alpha / (alpha + beta)
        lambda -> alpha + beta
        """
        super().__init__(
            t0=t0,
            x0_sub=x0,
            P0_sub=p0,
            process_noise_sub=process_noise,
            default_confidence=default_confidence,
            verbose=verbose,
        )

    def _propagate(self, dt: float):
        pass

    def _update(
        self,
        timestamp: float,
        trust: Union["UncertainTrustArray", "TrustArray"],
    ) -> None:
        pass


#############################################################
# FLOAT IMPLEMENTATIONS
#############################################################


@ALGORITHMS.register_module()
class VotingTrustEstimator(_TrustEstimator):
    multi = False

    def __init__(
        self,
        t0: float,
        p0: float = 0.5,
        time_window: Union[int, None] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(t0=t0, verbose=verbose)
        self.p0 = p0
        self.time_window = time_window
        self.trusted_buffer = PriorityQueue(max_size=None, max_heap=False)
        self.distrusted_buffer = PriorityQueue(max_size=None, max_heap=False)

    @property
    def n_votes(self) -> int:
        return self.n_trusted + self.n_distrusted

    @property
    def n_trusted(self) -> int:
        return len(self.trusted_buffer)

    @property
    def n_distrusted(self) -> int:
        return len(self.distrusted_buffer)

    @property
    def p(self):
        return self.mean()

    @property
    def q(self):
        return 1 - self.p

    def mean(self):
        if self.n_votes > 0:
            return self.binomial_test().statistic
        else:
            return self.p0

    def variance(self):
        return self.p * self.q

    def pdf(self, x: Union[float, np.ndarray]):
        p_out = self.p * np.ones((len(x),))
        p_out[x <= 0.5] = self.q
        return p_out

    def binomial_test(self) -> "BinomTestResult":
        result = binomtest(
            k=self.n_trusted, n=self.n_votes, p=self.p0, alternative="less"
        )
        return result

    def _propagate(self, dt: float) -> None:
        pass

    def _update(
        self, timestamp: float, trust: bool, threshold: float = 0.5, *args, **kwargs
    ):
        # flush old elements from the buffer
        if self.time_window:
            self.trusted_buffer.pop_all_below(priority_max=timestamp - self.time_window)
            self.distrusted_buffer.pop_all_below(
                priority_max=timestamp - self.time_window
            )

        # add the latest trust measurement
        trust = trust >= threshold
        if trust == 0.0:
            self.distrusted_buffer.push(priority=timestamp, item=trust)
        elif trust == 1.0:
            self.trusted_buffer.push(priority=timestamp, item=trust)
        else:
            raise ValueError(f"Trust must be boolean for voting, got {trust}")


@ALGORITHMS.register_module()
class MaximumLikelihoodTrustEstimator(_TrustEstimator):
    multi = False

    def __init__(
        self,
        t0: float = 0.0,
        distribution: ConfigDict = {"type": "Beta", "alpha": 1.0, "beta": 1.0},
        update_rate: float = 10,
        time_window: Union[int, None] = None,
        forgetting: float = 0.0,
        max_variance: float = None,
        n_min_update: int = 5,
        update_weighting: str = "uniform",
        verbose: bool = False,
    ) -> None:
        """Estimate distribution based on maximum likelihood over a window

        inputs:
            dist: the distribution to model the trust on
            update_rate: the rate at which we fit the model to the available data
            time_window: the amount of time back to use trust measurements
            forgetting: the amount of variance inflation to add during propagation (amt / second)
        """
        super().__init__(t0=t0, verbose=verbose)
        self.dist = MODELS.build(distribution)
        self.update_rate = update_rate
        self.time_window = time_window
        self.forgetting = forgetting
        self.variance_scaling = 1.0
        self.max_variance = max_variance
        self.n_min_update = n_min_update
        self.update_weighting = update_weighting
        self.buffer = PriorityQueue(max_size=None, max_heap=False)

    def mean(self):
        return self.dist.mean

    def variance(self):
        return self.dist.variance

    def pdf(self, x: Union[np.ndarray, float]):
        return self.dist.pdf(x=x)

    def sample(self, n: int) -> np.ndarray:
        return self.dist.rvs(n=n)

    def _propagate(self, dt: float) -> None:
        """Propagation model for the trust distribution

        TODO: propagation should interpolate back to uniform distribution
        TODO: figure out the phi, var of the uniform and interpolate (?)
        TODO: max variance should be that which comes of the uniform (?)
        """
        if self.forgetting != 0:
            self.variance_scaling = 1.0 + self.forgetting * dt
            if self.max_variance:
                var = min(self.max_variance, self.dist.var * self.variance_scaling)
            else:
                var = self.dist.var * self.variance_scaling
            self.dist.set_via_moments(self.dist.mean, var)

    def _update(self, timestamp: float, trust: float, *args, **kwargs):
        """Update model for the trust distribution"""
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
class ParticleFilter(_TrustEstimator):
    multi = False

    def __init__(
        self,
        t0: float = 0.0,
        n_particles: int = 100,
        n_eff_pct: float = 0.20,
        noise_variance: float = 0.5**2,
        prior: ConfigDict = {"type": "Uniform", "a": 0.0, "b": 1.0},
        likelihood: ConfigDict = {
            "type": "Normal",
            "mean": 0.0,
            "variance": 0.2**2,
        },
        sampler=systematic_sampling,
        n_steps_between_resampling: int = 0,
        verbose: bool = False,
    ) -> None:
        """Create a particle filter to estimate trust

        noise_variance: variance of the noise (s^2)
        prior: prior distribution of particles
        likelihood: p(y | x) --> observation likelihood
        TODO: likelihood should probably be difference between beta distributions

        Takes as input pseudomeasurements of the trust
        """
        super().__init__(t0=t0, verbose=verbose)
        self.prior = MODELS.build(prior)
        self.likelihood = MODELS.build(likelihood)
        self.sampler = sampler
        self.particles = self.prior.rvs(n=n_particles)
        self.weights = np.ones((n_particles,), dtype=float) / n_particles
        self.noise_sigma = np.sqrt(noise_variance)
        self.n_eff_pct = n_eff_pct
        self.i_steps_between_resampling = 0
        self.n_steps_between_resampling = n_steps_between_resampling

    @property
    def n_particles(self):
        return len(self.particles)

    @property
    def n_eff(self):
        return n_eff(self.weights)

    @property
    def mean(self) -> float:
        return sum(self.weights * self.particles)

    @property
    def variance(self) -> float:
        return sum(self.weights * (self.particles - self.mean) ** 2)

    def sample(self, n: int) -> np.ndarray:
        return self.particles[self.sampler(self.weights, n=n)]

    def check_and_resample(self) -> None:
        if self.i_steps_between_resampling > self.n_steps_between_resampling:
            if self.n_eff < self.n_eff_pct * self.n_particles:
                self.resample()
        self.i_steps_between_resampling = 0

    def resample(self) -> None:
        if self.verbose:
            print("resampling")
        self.particles = self.particles[self.sampler(self.weights, self.n_particles)]
        self.weights.fill(1.0 / self.n_particles)

    def _propagate(self, dt: float) -> None:
        """Propagate by adding noise to the particles

        To handle the boundary constraints, continue to sample
        noise until the particles are within constraints

        According to: "Bayesian estimation via sequential
        Monte Carlo sampling—Constrained dynamic systems"

        TODO: is this right?
        TODO: how to scale with dt?
        TODO: should we clip or bounce at the boundaries?
        """
        idx_sample = np.ones((self.n_particles,), dtype=bool)
        new_particles = np.copy(self.particles)
        while sum(idx_sample) > 0:
            noise = dt * self.noise_sigma * np.random.randn(sum(idx_sample))
            new_particles[idx_sample] = self.particles[idx_sample] + noise
            idx_sample = (new_particles < 0.0) | (1.0 < new_particles)
        self.particles = new_particles


@ALGORITHMS.register_module()
class SIRParticleFilterTrustEstimator(ParticleFilter):
    def _update(self, timestamp: float, trust: float) -> None:
        """Update particles for the trust distribution"""

        # update particles
        y = trust - self.particles  # innovation
        self.weights *= self.likelihood.pdf(y)
        self.weights += EPS  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize
        self.t_last_update = timestamp

        # resample particles if needed
        self.i_steps_between_resampling += 1
        self.check_and_resample()


@ALGORITHMS.register_module()
class PHDParticleFilterTrustEstimator(ParticleFilter):
    pass


@ALGORITHMS.register_module()
class GaussianSumTrustEstimator(ParticleFilter):
    pass
