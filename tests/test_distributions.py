from avtrust.distributions import TrustBetaDistribution
from avtrust.measurement import Psm


def test_copy_beta():
    beta = TrustBetaDistribution(timestamp=0, identifier=0, alpha=1.0, beta=1.0)
    b2 = beta.copy()
    assert beta.alpha == b2.alpha
    assert beta.beta == b2.beta


def test_init_beta():
    beta = TrustBetaDistribution(timestamp=0, identifier=0, alpha=1.0, beta=1.0)


def test_update_beta():
    beta = TrustBetaDistribution(timestamp=0, identifier=0, alpha=1.0, beta=1.0)
    psm = Psm(timestamp=0.0, target=0, value=1.0, confidence=1.0, source=1)
    beta.update(psm=psm)
    assert beta.alpha > 1.0
