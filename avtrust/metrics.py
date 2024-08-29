from typing import TYPE_CHECKING, Dict, NamedTuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avtrust.distributions import TrustDistribution

import numpy as np
from avstack.metrics import ConfusionMatrix
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .distributions import TrustArray


class AggregateAgentTrustMetric:
    def __init__(self, agent_metrics: Dict[int, "AgentTrustMetric"]):
        self.agent_metrics = agent_metrics

    def __getitem__(self, key: int) -> "AgentTrustMetric":
        return self.agent_metrics[key]

    @property
    def mean_metric(self):
        """Try to maximize this across all agents"""
        return np.mean([metric.metric for metric in self.agent_metrics.values()])


class AggregateTrackTrustMetric:
    def __init__(self, track_metrics: Dict[int, "TrackTrustMetric"]):
        self.track_metrics = track_metrics

    def __getitem__(self, key: int) -> "TrackTrustMetric":
        return self.track_metrics[key]

    @property
    def mean_metric(self):
        """Mix both good and bad tracks and try to maximize"""
        return np.mean([metric.metric for metric in self.track_metrics.values()])

    @property
    def mean_area_above_cdf_assigned(self):
        """For good tracks, try to maximize this"""
        return np.mean(
            [
                metric.metric
                for metric in self.track_metrics.values()
                if metric.assigned_to_truth
            ]
        )

    @property
    def mean_area_below_cdf_unassigned(self):
        """For bad tracks, to maximize this"""
        return np.mean(
            [
                metric.metric
                for metric in self.track_metrics.values()
                if not metric.assigned_to_truth
            ]
        )

    @property
    def n_assigned_tracks(self):
        """Number of assigned tracks"""
        return sum([metric.assigned_to_truth for metric in self.track_metrics.values()])

    @property
    def n_unassigned_tracks(self):
        """Number of unassigned tracks"""
        return sum(
            [not metric.assigned_to_truth for metric in self.track_metrics.values()]
        )


class AgentTrustMetric(NamedTuple):
    """Container for metric on agent trust

    f1_score: float --> good agent maximizes this
    area_above_cdf: float --> area based on trust cdf curve
    f1_threshold: float --> threshold on what is a "trustworthy"

    if f1 < threshold, call the agent untrustworthy, so try
    to maximize area below the CDF. if f1 >= threshold, call
    the agent trustworthy, so try to maximize the area above CDF
    """

    f1_score: float
    area_above_cdf: float
    f1_threshold: float = 0.9

    @property
    def metric(self) -> float:
        """Try to maximize this metric. Max value is 1.0"""
        return (
            self.area_above_cdf
            if self.f1_score >= self.f1_threshold
            else 1 - self.area_above_cdf
        )


class TrackTrustMetric(NamedTuple):
    """Container for metrics on a single track trust

    area_above_cdf: float
    assigned_to_truth: bool

    if assigned to truth, try to maximize area above
    cdf, if not assinged to truth, try to maximize area
    below the cdf
    """

    area_above_cdf: float
    assigned_to_truth: bool

    @property
    def metric(self) -> float:
        return (
            self.area_above_cdf if self.assigned_to_truth else 1 - self.area_above_cdf
        )


def area_below_cdf(trust: "TrustDistribution") -> float:
    dx = 1.0 / 1000
    xs = np.arange(start=0, stop=1.0, step=dx)
    cdfs = trust.cdf(xs)
    return sum(cdfs * dx)


def area_above_cdf(trust: "TrustDistribution") -> float:
    return 1 - area_below_cdf(trust)


def get_trust_agents_metrics(
    truths_agent: Dict[int, "DataContainer"],
    tracks_agent: Dict[int, "DataContainer"],
    trust_agents: "TrustArray",
    assign_radius: float = 2.0,
    f1_threshold: float = 0.9,
) -> AggregateAgentTrustMetric:
    """Get metrics for the trust scores on agents"""

    # assign agent tracks to agent viewable truths
    agent_metrics = {}
    for ID_agent in truths_agent:
        A_truth = build_A_from_distance(tracks_agent[ID_agent], truths_agent[ID_agent])
        assigns = gnn_single_frame_assign(A_truth, cost_threshold=assign_radius)

        # build confusion matrix
        confusion = ConfusionMatrix(
            n_true_positives=len(assigns),
            n_true_negatives=0,  # because assignment problem
            n_false_positives=len(assigns.unassigned_rows),
            n_false_negatives=len(assigns.unassigned_cols),
        )

        # get the metric
        agent_metrics[ID_agent] = AgentTrustMetric(
            f1_score=confusion.f1_score,
            area_above_cdf=area_above_cdf(trust_agents[ID_agent]),
            f1_threshold=f1_threshold,
        )

    return AggregateAgentTrustMetric(agent_metrics)


def get_trust_tracks_metrics(
    truths: "DataContainer",
    tracks_cc: "DataContainer",
    trust_tracks: "TrustArray",
    assign_radius: float = 2.0,
) -> AggregateTrackTrustMetric:
    """Get metrics for the trust scores on tracks"""

    # assign cc tracks to truths
    A = build_A_from_distance(tracks_cc, truths)
    assigns = gnn_single_frame_assign(A, cost_threshold=assign_radius)
    IDs_assign = [tracks_cc[j_track].ID for j_track in assigns._row_to_col]

    # get the metric for each track
    track_metrics = {}
    for track in tracks_cc:
        track_metrics[track.ID] = TrackTrustMetric(
            area_above_cdf=area_above_cdf(trust_tracks[track.ID]),
            assigned_to_truth=(track.ID in IDs_assign),
        )

    return AggregateTrackTrustMetric(track_metrics)
