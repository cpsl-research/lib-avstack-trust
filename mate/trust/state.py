
class _TrustInformedWrapper:
    def __init__(self, element, estimator) -> None:
        self.element = element
        self.estimator = estimator

    def propagate(self, timestamp: float):
        self.estimator.propagate(timestamp)

    def update(self, timestamp: float, trust: float):
        self.estimator.update(timestamp, trust)


class TrustInformedCluster(_TrustInformedWrapper):
    pass


class TrustInformedAgent(_TrustInformedWrapper):
    pass