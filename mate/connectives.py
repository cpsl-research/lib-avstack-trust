from typing import List, Union

import numpy as np
from avstack.config import MODELS, ConfigDict

from .measurement import UncertainTrustFloat


EPS = np.finfo(float).eps


def normalize(weight: np.ndarray) -> np.ndarray:
    return weight / sum(weight)


##############################################
# OPERATORS
##############################################


class Operator:
    @staticmethod
    def conjunction(a: float, b: float) -> float:
        raise NotImplementedError

    @staticmethod
    def disjunction(a: float, b: float) -> float:
        raise NotImplementedError

    @staticmethod
    def negation(a: float) -> float:
        raise NotImplementedError


@MODELS.register_module()
class CrispOperator(Operator):
    @staticmethod
    def conjunction(a: bool, b: bool) -> bool:
        assert isinstance(a, bool)
        assert isinstance(b, bool)
        return a and b

    @staticmethod
    def disjunction(a: bool, b: bool) -> bool:
        assert isinstance(a, bool)
        assert isinstance(b, bool)
        return a or b

    @staticmethod
    def negation(a: bool) -> bool:
        return not a


@MODELS.register_module()
class ZadehOperator(Operator):
    @staticmethod
    def conjunction(a: float, b: float) -> float:
        return min(a, b)

    @staticmethod
    def disjunction(a: float, b: float) -> float:
        return max(a, b)

    @staticmethod
    def negation(a: float) -> float:
        return 1 - a


@MODELS.register_module()
class ProbabilityOperator(Operator):
    @staticmethod
    def conjunction(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def disjunction(a: float, b: float) -> float:
        return a + b - a * b

    @staticmethod
    def negation(a: float) -> float:
        return 1 - a


@MODELS.register_module()
class EinsteinOperator(Operator):
    @staticmethod
    def conjunction(a: float, b: float) -> float:
        return a * b / (1 + (1 - a) * (1 - b))

    @staticmethod
    def disjunction(a: float, b: float) -> float:
        return (a + b) / (1 + a * b)

    @staticmethod
    def negation(a: float) -> float:
        return 1 - a


@MODELS.register_module()
class BoundaryOperators(Operator):
    @staticmethod
    def conjunction(a: float, b: float) -> float:
        return max(0, a + b - 1)

    @staticmethod
    def disjunction(a: float, b: float) -> float:
        return min(1, a + b)

    @staticmethod
    def negation(a: float) -> float:
        return 1 - a


@MODELS.register_module()
class CertainTrustOperators(Operator):
    """Model that includes uncertainty in trust

    According to: CertainLogic: A Logic for Modeling Trust and Uncertainty

    t: trust value
    c: certainty
    f: prior/default
    """

    @staticmethod
    def conjunction(
        a_tcf: "UncertainTrustFloat", b_tcf: "UncertainTrustFloat"
    ) -> "UncertainTrustFloat":
        ta, ca, fa, _ = a_tcf
        tb, cb, fb, _ = b_tcf
        c = (
            ca
            + cb
            - ca * cb
            - ((1 - ca) * cb * (1 - fa) * tb + ca * (1 - cb) * (1 - fb) * ta)
            / (1 - fa * fb)
        )
        if c > 0 + EPS:
            t = (
                1
                / c
                * (
                    ca * cb * ta * tb
                    + (
                        ca * (1 - cb) * (1 - fa) * fb * ta
                        + (1 - ca) * cb * fa * (1 - fb) * tb
                    )
                    / (1 - fa * fb)
                )
            )
        else:
            t = 0.5
        f = fa * fb
        return UncertainTrustFloat(t, c, f, ID=None)

    @staticmethod
    def disjunction(
        a_tcf: "UncertainTrustFloat", b_tcf: "UncertainTrustFloat"
    ) -> "UncertainTrustFloat":
        ta, ca, fa, _ = a_tcf
        tb, cb, fb, _ = b_tcf
        c = (
            ca
            + cb
            - ca * cb
            - (
                (ca * (1 - cb) * fb * (1 - ta) + (1 - ca) * cb * fa * (1 - tb))
                / (fa + fb - fa * fb)
            )
        )
        if c > 0 + EPS:
            t = 1 / c * (ca * ta + cb * tb - ca * cb * ta * tb)
        else:
            t = 0.5
        f = fa + fb - fa * fb
        return UncertainTrustFloat(t, c, f, ID=None)

    @staticmethod
    def negation(a_tcf: "UncertainTrustFloat") -> "UncertainTrustFloat":
        ta, ca, fa, _ = a_tcf
        return UncertainTrustFloat(1 - ta, ca, 1 - fa, ID=None)


##############################################
# TRIPLES
##############################################


class _DeMorganTriple:
    """Abstract base class for a DeMorgan triple connective"""

    def norm(self, inputs: List[Union[float, bool]]) -> Union[float, bool]:
        """Apply all the conjunction operations

        Recursive application is valid for T-norms
        """
        out = inputs[0]
        for inp in inputs[1:]:
            out = self.operator.conjunction(out, inp)
        return out

    def conorm(self, inputs: List[Union[float, bool]]) -> Union[float, bool]:
        """Apply all the disjunction operations

        Recursive application is valid for T-norms
        """
        out = inputs[0]
        for inp in inputs[1:]:
            out = self.operator.disjunction(out, inp)
        return out

    def negation(
        inputs: Union[List[float], List[bool]]
    ) -> Union[List[float], List[bool]]:
        raise NotImplementedError


@MODELS.register_module()
class StandardCrisp(_DeMorganTriple):
    operator = CrispOperator

    @staticmethod
    def negation(inputs: List[bool]) -> bool:
        return [~ai for ai in inputs]


@MODELS.register_module()
class StandardFuzzy(_DeMorganTriple):
    def __init__(self, operator: ConfigDict):
        self.operator = MODELS.build(operator)

    @staticmethod
    def negation(inputs: List[float]) -> List[float]:
        return [1 - ai for ai in inputs]


####################################################
# WEIGHTED MODELS
####################################################


@MODELS.register_module()
class WeightedAverageFuzzy(_DeMorganTriple):
    """Simple weighted average assuming weights sum to 1

    This suffers from two problems, according to
    "The Weighting Issue in Fuzzy Logic"
    (1) Degeneracy problem -- does not reduce to unweighted form
    (2) Differential problem -- conjunction and disjunction are the same
    """

    @staticmethod
    def norm(inputs: List[float], weight: np.ndarray) -> float:
        return np.prod(np.asarray(inputs) * normalize(weight))

    @staticmethod
    def conorm(inputs: List[float], weight: np.ndarray) -> float:
        return np.prod(np.asarray(inputs) * normalize(weight))

    @staticmethod
    def negation(inputs: List[float]) -> List[float]:
        return [1 - ai for ai in inputs]


@MODELS.register_module()
class CaiWeightedFuzzy(_DeMorganTriple):
    """Weighted fuzzy logic according to Cai et al.
        "The Weighting Issue in Fuzzy Logic"

    Model has the following properties:
    (1) Model reduces to non-weighted in case of uniform weights
    (2) Conjunction and disjunction are separate
    (3) Every piece of information is considered in some way
    """
