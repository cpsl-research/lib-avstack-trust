import json
import math
from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from .measurement import Psm
    from .propagator import DistributionPropagator

from avtrust.config import AVTRUST


class TrustDistEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, TrustBetaDistribution):
            trust_dict = {
                "alpha": o.alpha,
                "beta": o.beta,
            }
            return {"trustbetadist": trust_dict}
        else:
            raise NotImplementedError(f"{type(o)}, {o}")


class TrustDistDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "trustbetadist" in json_object:
            json_object = json_object["trustbetadist"]
            return TrustBetaDistribution(
                alpha=json_object["alpha"],
                beta=json_object["beta"],
            )
        else:
            return json_object


class TrustDistribution:
    @property
    def std(self):
        return math.sqrt(self.variance)

    def encode(self):
        return json.dumps(self, cls=TrustDistEncoder)

    def update(self, psm: "Psm"):
        raise NotImplementedError


@AVTRUST.register_module()
class TrustBetaDistribution(TrustDistribution):
    def __init__(
        self,
        timestamp: float,
        identifier: str,
        alpha: float,
        beta: float,
        negativity_bias: float = 2.0,
    ):
        self.timestamp = timestamp
        self.identifier = identifier
        self.alpha = alpha
        self.beta = beta
        self._negativity_bias = negativity_bias
        self._t_last_update = timestamp

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"TrustBetaDistribution: ({self.alpha:5.2f}, {self.beta:5.2f})"

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        if alpha <= 0:
            raise ValueError("Alpha must be larger than 0")
        self._alpha = alpha

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta: float):
        if beta <= 0:
            raise ValueError("Beta must be larger than 0")
        self._beta = beta

    @property
    def a(self):
        return self.alpha

    @a.setter
    def a(self, alpha):
        self.alpha = alpha

    @property
    def b(self):
        return self.beta

    @b.setter
    def b(self, beta):
        self.beta = beta

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def precision(self):
        return self.alpha + self.beta

    @property
    def dt_last_update(self):
        return self.timestamp - self._t_last_update

    @property
    def variance(self):
        return (
            self.alpha
            * self.beta
            / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        )

    def copy(self):
        return TrustBetaDistribution(
            timestamp=self.timestamp,
            identifier=self.identifier,
            alpha=self.alpha,
            beta=self.beta,
        )

    def update(self, psm: "Psm"):
        if psm.target != self.identifier:
            raise ValueError(
                f"PSM {psm.target} target does not match trust identifier {self.identifier}"
            )
        n = self._negativity_bias
        w_pos = 2 / (n + 1)
        w_neg = 2 * n / (n + 1)
        self.alpha += w_pos * psm.confidence * psm.value
        self.beta += w_neg * psm.confidence * (1 - psm.value)
        self._t_last_update = psm.timestamp


class TrustArray:
    def __init__(self, timestamp: float, trusts: List[TrustDistribution]):
        self.timestamp = timestamp
        self.trusts = {tr.identifier: tr for tr in trusts}

    def __iter__(self):
        return iter(self.trusts)

    def __getitem__(self, key: int) -> TrustDistribution:
        return self.trusts[key]

    def __len__(self) -> int:
        return len(self.trusts)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"TrustArray at time {self.timestamp}, {self.trusts}"

    def keys(self):
        return self.trusts.keys()

    def append(self, other: "TrustDistribution"):
        self.trusts[other.identifier] = other

    def propagate(self, timestamp: float, propagator: "DistributionPropagator"):
        for trust in self.trusts.values():
            propagator.propagate(timestamp, trust)
        self.timestamp = timestamp

    def remove(self, other: "TrustDistribution"):
        del self.trusts[other.identifier]
