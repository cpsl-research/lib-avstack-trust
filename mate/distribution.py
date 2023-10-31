import numpy as np
from scipy import stats


class _Distribution:
    @property
    def var(self):
        return self.variance
    
    @staticmethod
    def fit(x: list | np.ndarray):
        raise NotImplementedError

    def set_via_moments(self, mean: float, variance: float):
        raise NotImplementedError

    def pdf(self, x: float):
        raise NotImplementedError

    def cdf(self, x: float):
        raise NotImplementedError

    def rvs(self, n: int):
        raise NotImplementedError

    def partial(self, param: str):
        raise NotImplementedError


class Beta(_Distribution):
    def __init__(
        self,
        alpha: float = None,
        beta: float = None,
        phi: float = None,
        lam: float = None,
        mean: float = None,
        var: float = None,
    ) -> None:
        if (alpha is not None) and (beta is not None):
            self.alpha = alpha
            self.beta = beta
        elif (phi is not None) and (lam is not None):
            self.set_via_phi_lam(phi, lam)
        elif (mean is not None) and (var is not None):
            self.set_via_moments(mean, var)
        else:
            raise ValueError(
                "Cannot understand input combination. Either needs to be parameterized by (alpha, beta), (phi, lam), or (mean, var)"
            )

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0")
        self._alpha = alpha

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta <= 0:
            raise ValueError("Beta must be greater than 0")
        self._beta = beta

    @property
    def a(self):
        return self.alpha

    @property
    def b(self):
        return self.beta

    @property
    def phi(self):
        return self.a / (self.a + self.b)

    @property
    def lam(self):
        return self.a + self.b

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    @property
    def variance(self):
        return self.a * self.b / ((self.a + self.b) ** 2 * (self.a + self.b + 1))

    @staticmethod
    def fit(x: list | np.ndarray):
        a, b, _, _ = stats.beta.fit(x)
        return Beta(alpha=a, beta=b)

    def set_via_moments(self, mean: float, variance: float):
        c0 = (mean * (1 - mean) / variance) - 1
        self.alpha = c0 * mean
        self.beta = c0 * (1 - mean)

    def set_via_phi_lam(self, phi: float, lam: float):
        self.alpha = phi * lam
        self.beta = (1 - phi) * lam

    def pdf(self, x: float):
        return stats.beta.pdf(x, self.a, self.b)

    def cdf(self, x: float):
        return stats.beta.cdf(x, self.a, self.b)
    
    def rvs(self, n: int, random_state: int|None = None):
        return stats.beta.rvs(self.a, self.b, size=n, random_state=random_state)

    def partial(self, param: str):
        if param == "alpha":
            raise
        elif param == "beta":
            raise
        elif param == "phi":
            raise
        elif param == "lam":
            raise
        else:
            raise NotImplementedError(param)


class OddsBeta(Beta):
    "Also called beta prime distribution"

    @property
    def mean(self):
        if self.b > 1:
            return self.a / (self.b - 1)
        else:
            raise OverflowError("Mean is undefined for beta less than 1")

    @property
    def variance(self):
        if self.b > 2:
            return self.a * (self.a + self.b - 1) / ((self.b - 2) * (self.b - 1) ** 2)
        else:
            raise OverflowError("Variance is undefined for beta less than 2")

    @staticmethod
    def fit(x: list | np.ndarray):
        a, b, _, _ = stats.betaprime.fit(x)
        return OddsBeta(alpha=a, beta=b)

    def set_via_moments(self, mean: float, variance: float):
        raise NotImplementedError("Cannot set via moments for beta prime distribution.")

    def pdf(self, x: float):
        return stats.betaprime.pdf(x, self.a, self.b)

    def cdf(self, x: float):
        return stats.betaprime.cdf(x, self.b, self.b)
        
    def rvs(self, n: int, random_state: int|None = None):
        return stats.betaprime.rvs(self.a, self.b, size=n, random_state=random_state)

    def partial(self, param: str):
        if param == "alpha":
            raise
        elif param == "beta":
            raise
        elif param == "phi":
            raise
        elif param == "lam":
            raise
        else:
            raise NotImplementedError(param)


class LogBeta(Beta):
    pass


class LogOddsBeta(OddsBeta):
    pass


class GaussianMixture(_Distribution):
    pass
