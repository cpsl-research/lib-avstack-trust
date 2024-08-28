from typing import NamedTuple

import numpy as np

from avtrust.distributions import TrustArray, TrustBetaDistribution
from avtrust.metrics import area_above_cdf, area_below_cdf, get_trust_tracks_metrics


def test_area_uniform():
    dist = TrustBetaDistribution(
        timestamp=0,
        identifier=0,
        alpha=1,
        beta=1,
    )
    assert np.isclose(area_below_cdf(dist), 0.5, atol=0.001)
    assert np.isclose(area_above_cdf(dist), 0.5, atol=0.001)


def test_area_below_beta():
    dist = TrustBetaDistribution(
        timestamp=0,
        identifier=0,
        alpha=5,
        beta=1,
    )
    assert area_below_cdf(dist) < 0.4


class TestObject(NamedTuple):
    x: np.ndarray
    ID: int

    def distance(self, other):
        return np.linalg.norm(self.x - other.x)


def test_get_trust_tracks_metrics_all_assigned():
    # get some truths and tracks
    seed = 0
    rng = np.random.RandomState(seed=seed)
    n_truths = 10
    n_fn = 2
    n_fp = 1
    truths = [TestObject(x=100 * rng.randn(3), ID=i) for i in range(n_truths)]
    truths.extend(
        [TestObject(x=100 * rng.randn(3), ID=i + n_truths) for i in range(n_fn)]
    )
    tracks_cc = [TestObject(truth.x, truth.ID) for truth in truths[:n_truths]]
    tracks_cc.extend(
        [TestObject(x=100 * rng.randn(3), ID=i + n_truths) for i in range(n_fp)]
    )

    # trust for the correct assignments
    trust_tracks = TrustArray(
        timestamp=0.0,
        trusts=[
            TrustBetaDistribution(
                timestamp=0,
                identifier=i,
                alpha=6,
                beta=1,
            )
            for i in range(n_truths)
        ],
    )

    # trust for the incorrect assignments
    trust_tracks.extend(
        TrustArray(
            timestamp=0.0,
            trusts=[
                TrustBetaDistribution(
                    timestamp=0.0,
                    identifier=i + n_truths,
                    alpha=1,
                    beta=6,
                )
                for i in range(n_fp)
            ],
        )
    )

    # run the metrics evaluation
    metrics = get_trust_tracks_metrics(
        truths=truths, tracks_cc=tracks_cc, trust_tracks=trust_tracks, assign_radius=2.0
    )

    # this should result in small maba
    assert metrics.m_area_below_assigned < 0.2
    assert metrics.m_area_above_unassigned < 0.2
    assert metrics.n_unassigned_truths == n_fn
    assert not np.isnan(metrics.m_area_above_unassigned)
