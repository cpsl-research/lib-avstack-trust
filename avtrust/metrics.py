from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avtrust.distributions import TrustDistribution

import numpy as np
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .distributions import TrustArray


class MetricsAgentTrust(NamedTuple):
    """Container for metrics on agent trust"""


class MetricsTrackTrust(NamedTuple):
    """Container for metrics on track trust"""

    m_area_below_assigned: float
    m_area_above_unassigned: float
    n_unassigned_truths: int


def area_below_cdf(trust: "TrustDistribution") -> float:
    dx = 1.0 / 1000
    xs = np.arange(start=0, stop=1.0, step=dx)
    cdfs = trust.cdf(xs)
    return sum(cdfs * dx)


def area_above_cdf(trust: "TrustDistribution") -> float:
    return 1 - area_below_cdf(trust)


def get_trust_agents_metrics(
    truths: "DataContainer",
    tracks_agent: "DataContainer",
    tracks_cc: "DataContainer",
    trust_agents: "TrustArray",
    trust_tracks: "TrustArray",
    assign_radius: float = 2.0,
) -> MetricsAgentTrust:
    """Get metrics for the trust scores on agents"""

    # assign agent tracks to truths
    A_truth = build_A_from_distance(tracks_agent, truths)
    assigns_truth = gnn_single_frame_assign(A_truth, cost_threshold=assign_radius)

    # assign agent tracks to cc
    A_cc = build_A_from_distance(tracks_agent, tracks_cc)
    assigns_cc = gnn_single_frame_assign(A_cc, cost_threshold=assign_radius)

    # (1) check trust on agent tracks assigned to cc

    # (2) tracks in agents not assign to truths

    # (3) penalize for unassigned truths
    # TODO: don't know how to do this properly yet....
    return None


def get_trust_tracks_metrics(
    truths: "DataContainer",
    tracks_cc: "DataContainer",
    trust_tracks: "TrustArray",
    assign_radius: float = 2.0,
) -> MetricsTrackTrust:
    """Get metrics for the trust scores on tracks"""

    # assign cc tracks to truths
    A = build_A_from_distance(tracks_cc, truths)
    assigns = gnn_single_frame_assign(A, cost_threshold=assign_radius)

    # get the tracks that were assigned to truths
    IDs_assign = [tracks_cc[j_track].ID for j_track in assigns._row_to_col]
    IDs_unassign = [tracks_cc[j_track].ID for j_track in assigns.unassigned_rows]

    # (1) check the trust of the assigned tracks
    area_below_assigned = [area_below_cdf(trust_tracks[ID]) for ID in IDs_assign]

    # (2) check the trust of the unassigned tracks
    area_above_unassigned = [area_above_cdf(trust_tracks[ID]) for ID in IDs_unassign]

    # (3) penalize for unassigned truths
    n_unassigned_truths = len(assigns.unassigned_cols)

    return MetricsTrackTrust(
        m_area_below_assigned=np.mean(area_below_assigned),
        m_area_above_unassigned=np.mean(area_above_unassigned),
        n_unassigned_truths=n_unassigned_truths,
    )
