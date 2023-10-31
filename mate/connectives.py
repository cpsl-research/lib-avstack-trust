class DeMorganTriple:
    """Abstract base class for a DeMorgan triple connective"""

    @staticmethod
    def norm(a, b):
        raise NotImplementedError

    @staticmethod
    def conorm(a, b):
        raise NotImplementedError

    @staticmethod
    def negation(a):
        raise NotImplementedError


class StandardCrisp(DeMorganTriple):
    @staticmethod
    def norm(a: bool, b: bool) -> bool:
        return a and b

    @staticmethod
    def conorm(a: bool, b: bool) -> bool:
        return a or b

    @staticmethod
    def negation(a: bool) -> bool:
        return not a


class StandardFuzzy(DeMorganTriple):
    @staticmethod
    def norm(a: float, b: float) -> float:
        return min(a, b)

    @staticmethod
    def conorm(a: float, b: float) -> float:
        return max(a, b)

    @staticmethod
    def negation(a: float) -> float:
        return 1 - a
