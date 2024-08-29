from typing import NamedTuple

import numpy as np

from avtrust.distributions import TrustArray, TrustBetaDistribution
from avtrust.metrics import (
    area_above_cdf,
    area_below_cdf,
    get_trust_agents_metrics,
    get_trust_tracks_metrics,
)


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


def _get_truths_and_tracks(rng, n_truths, n_fn, n_fp):
    truths = [TestObject(x=100 * rng.randn(3), ID=i) for i in range(n_truths)]
    truths.extend(
        [TestObject(x=100 * rng.randn(3), ID=i + n_truths) for i in range(n_fn)]
    )
    tracks = [TestObject(truth.x, truth.ID) for truth in truths[:n_truths]]
    tracks.extend(
        [TestObject(x=100 * rng.randn(3), ID=i + n_truths) for i in range(n_fp)]
    )
    return truths, tracks


def test_get_trust_tracks_metrics():
    # get some truths and tracks
    seed = 0
    rng = np.random.RandomState(seed=seed)
    n_truths = 10
    n_fn = 2
    n_fp = 1
    truths, tracks_cc = _get_truths_and_tracks(rng, n_truths, n_fn, n_fp)

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
    track_metrics = get_trust_tracks_metrics(
        truths=truths, tracks_cc=tracks_cc, trust_tracks=trust_tracks, assign_radius=2.0
    )

    # this should result in small maba
    assert track_metrics.mean_area_above_cdf_assigned > 0.8
    assert track_metrics.mean_area_below_cdf_unassigned > 0.8
    assert track_metrics.n_assigned_tracks == n_truths
    assert track_metrics.n_unassigned_tracks == n_fp


def test_get_agent_tracks_metrics():
    # get some truths and tracks
    seed = 0
    rng = np.random.RandomState(seed=seed)
    n_truths = 10
    n_fn = {0: 0, 1: 0, 2: 4, 3: 3}
    n_fp = {0: 0, 1: 4, 2: 0, 3: 3}
    n_agents = len(n_fn)
    trust_agents = TrustArray(
        timestamp=0.0,
        trusts=[
            TrustBetaDistribution(
                timestamp=0,
                identifier=i,
                alpha=6,
                beta=1,
            )
            for i in range(n_agents)
        ],
    )

    # build tracks for each agent
    truths_agents = {}
    tracks_agents = {}
    for ID_agent in range(n_agents):
        truths_agents[ID_agent], tracks_agents[ID_agent] = _get_truths_and_tracks(
            rng, n_truths, n_fn[ID_agent], n_fp[ID_agent]
        )

    # compute metrics
    agent_metrics = get_trust_agents_metrics(truths_agents, tracks_agents, trust_agents)

    # only agent 0 should have a good overall metric
    assert agent_metrics[0].metric > 0.7
    assert all([agent_metrics[ID_agent].metric < 0.4 for ID_agent in [1, 2, 3]])
