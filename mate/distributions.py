from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .measurement import PSM

import math

from mate.config import MATE


class TrustDistribution:
    @property
    def std(self):
        return math.sqrt(self.variance)

    def update(self, psm: "PSM"):
        raise NotImplementedError


@MATE.register_module()
class TrustBetaDistribution(TrustDistribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"TrustBetaDistribution: ({self.alpha:5.2f}, {self.beta:5.2f})"

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def precision(self):
        return self.alpha + self.beta

    @property
    def variance(self):
        return (
            self.alpha
            * self.beta
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )

    def copy(self):
        return TrustBetaDistribution(self.alpha, self.beta)

    def update(self, psm: "PSM"):
        self.alpha += psm.confidence * psm.value
        self.beta += psm.confidence * (1 - psm.value)
