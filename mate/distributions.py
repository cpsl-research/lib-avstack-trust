class TrustBetaParams:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"TrustBetaParams: ({self.alpha:5.2f}, {self.beta:5.2f})"

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
        return TrustBetaParams(self.alpha, self.beta)
