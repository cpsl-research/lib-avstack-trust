import json
import math
from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from .measurement import Psm
    from .propagator import DistributionPropagator

from mate.config import MATE


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


@MATE.register_module()
class TrustBetaDistribution(TrustDistribution):
    def __init__(self, timestamp: float, identifier: str, alpha: float, beta: float):
        self.timestamp = timestamp
        self.identifier = identifier
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

    def update(self, psm: "Psm"):
        self.alpha += psm.confidence * psm.value
        self.beta += psm.confidence * (1 - psm.value)


class TrustArray:
    def __init__(self, timestamp: float, trusts: List[TrustDistribution]):
        self.timestamp = timestamp
        self.trusts = {tr.identifier: tr for tr in trusts}

    def __iter__(self):
        return iter(self.trusts)

    def __getitem__(self, key: int) -> TrustDistribution:
        return self.trusts[key]
    
    def __len__(self) -> int:
        return len(self.psms)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"TrustArray at time {self.timestamp}, {self.trusts}"

    def append(self, other: "TrustDistribution"):
        self.trusts[other.identifier] = other

    def propagate(self, timestamp: float, propagator: "DistributionPropagator"):
        for trust in self.trusts.values():
            propagator.propagate(timestamp, trust)
        self.timestamp = timestamp
