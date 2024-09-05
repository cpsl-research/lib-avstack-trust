from typing import TYPE_CHECKING, Dict, NamedTuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avtrust.distributions import TrustDistribution

import json

import numpy as np
from avstack.metrics import ConfusionMatrix
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .distributions import TrustArray


class AggregateMetricsEncoder(json.JSONEncoder):
    def default(self, o):
        metrics_dict = {
            "keys": list(o.keys()),
            "values": [met.encode() for met in o.values()],
            "type": "agent" if isinstance(o, AggregateAgentTrustMetric) else "track",
        }
        return {"aggregate_metrics": metrics_dict}


class AggregateMetricsDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "aggregate_metrics" in json_object:
            json_object = json_object["aggregate_metrics"]
            metrics = {
                k: json.loads(v, cls=MetricDecoder)
                for k, v in zip(json_object["keys"], json_object["values"])
            }
            if metrics["type"] == "agent":
                return AggregateAgentTrustMetric(metrics)
            elif metrics["type"] == "track":
                return AggregateTrackTrustMetric(metrics)
            else:
                raise NotImplementedError(metrics["type"])
        return json_object


class MetricEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AgentTrustMetric):
            metric_dict = {
                "f1_score": o.f1_score,
                "area_above_cdf": o.area_above_cdf,
                "f1_threshold": o.f1_threshold,
            }
        elif isinstance(o, TrackTrustMetric):
            metric_dict = {
                "area_above_cdf": o.area_above_cdf,
                "assigned_to_truth": o.assigned_to_truth,
            }
        else:
            raise NotImplementedError(type(o))
        return {"metric": metric_dict}


class MetricDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "metric" in json_object:
            json_object = json_object["metric"]
            if json_object["type"] == "agent":
                return AgentTrustMetric(
                    f1_score=json_object["f1_score"],
                    area_above_cdf=json_object["area_above_cdf"],
                    f1_threshold=json_object["f1_threshold"],
                )
            elif json_object["type"] == "track":
                return TrackTrustMetric(
                    area_above_cdf=json_object["area_above_cdf"],
                    assigned_to_truth=json_object["assigned_to_truth"],
                )
            else:
                raise NotImplementedError(json_object["type"])
        return json_object


class _AggregateMetric:
    def __init__(self, timestamp: float, data_dict: Dict[int, "_Metric"]):
        self.timestamp = timestamp
        self.data_dict = data_dict

    def keys(self):
        return self.data_dict.keys()

    def values(self):
        return self.data_dict.values()

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key: int) -> "_Metric":
        return self.data_dict[key]

    @property
    def mean_metric(self):
        """Try to maximize this across all agents"""
        return np.nanmean([metric.metric for metric in self.data_dict.values()])

    def encode(self):
        return json.dumps(self, cls=AggregateMetricsEncoder)


class AggregateAgentTrustMetric(_AggregateMetric):
    def __init__(self, timestamp: float, agent_metrics: Dict[int, "AgentTrustMetric"]):
        super().__init__(timestamp, agent_metrics)

    def __str__(self) -> str:
        return f"AggregateAgentTrustMetric for {list(self.data_dict.keys())} agents with mean metric {self.mean_metric}"

    @property
    def agent_metrics(self):
        return self.data_dict


class AggregateTrackTrustMetric(_AggregateMetric):
    def __init__(self, timestamp: float, track_metrics: Dict[int, "TrackTrustMetric"]):
        super().__init__(timestamp, track_metrics)

    def __str__(self) -> str:
        return f"AggregateTrackTrustMetric for {list(self.data_dict.keys())} tracks with mean metric {self.mean_metric}"

    @property
    def track_metrics(self):
        return self.data_dict

    @property
    def mean_area_above_cdf_assigned(self):
        """For good tracks, try to maximize this"""
        return np.nanmean(
            [
                metric.metric
                for metric in self.data_dict.values()
                if metric.assigned_to_truth
            ]
        )

    @property
    def mean_area_below_cdf_unassigned(self):
        """For bad tracks, to maximize this"""
        return np.nanmean(
            [
                metric.metric
                for metric in self.data_dict.values()
                if not metric.assigned_to_truth
            ]
        )

    @property
    def n_assigned_tracks(self):
        """Number of assigned tracks"""
        return sum([metric.assigned_to_truth for metric in self.data_dict.values()])

    @property
    def n_unassigned_tracks(self):
        """Number of unassigned tracks"""
        return sum([not metric.assigned_to_truth for metric in self.data_dict.values()])


class _Metric(NamedTuple):
    pass


class AgentTrustMetric(NamedTuple):
    """Container for metric on agent trust

    f1_score: float --> good agent maximizes this
    area_above_cdf: float --> area based on trust cdf curve
    f1_threshold: float --> threshold on what is a "trustworthy"

    if f1 < threshold, call the agent untrustworthy, so try
    to maximize area below the CDF. if f1 >= threshold, call
    the agent trustworthy, so try to maximize the area above CDF
    """

    identifier: int
    timestamp: float
    f1_score: float
    area_above_cdf: float
    agent_is_attacked: bool
    f1_threshold: float = 0.9

    @property
    def metric(self) -> float:
        """Try to maximize this metric. Max value is 1.0"""
        # if not np.isnan(self.f1_score):
        #     return (
        #         self.area_above_cdf
        #         if self.f1_score >= self.f1_threshold
        #         else 1 - self.area_above_cdf
        #     )
        # else:
        #     return np.nan
        return (
            1 - self.area_above_cdf if self.agent_is_attacked else self.area_above_cdf
        )


class TrackTrustMetric(NamedTuple):
    """Container for metrics on a single track trust

    area_above_cdf: float
    assigned_to_truth: bool

    if assigned to truth, try to maximize area above
    cdf, if not assinged to truth, try to maximize area
    below the cdf
    """

    identifier: int
    timestamp: float
    area_above_cdf: float
    assigned_to_truth: bool

    @property
    def metric(self) -> float:
        return (
            self.area_above_cdf if self.assigned_to_truth else 1 - self.area_above_cdf
        )


def area_below_cdf(trust: "TrustDistribution") -> float:
    """This is just 1 - expectation

    The long form of doing it would be
        dx = 1.0 / 1000
        xs = np.arange(start=0, stop=1.0, step=dx)
        cdfs = trust.cdf(xs)
        return sum(cdfs * dx)
    """
    return 1 - trust.mean


def area_above_cdf(trust: "TrustDistribution") -> float:
    return 1 - area_below_cdf(trust)


def get_trust_agents_metrics(
    truths_agents: Dict[int, "DataContainer"],
    tracks_agents: Dict[int, "DataContainer"],
    trust_agents: "TrustArray",
    attacked_agents: set,
    assign_radius: float = 2.0,
    f1_threshold: float = 0.9,
) -> AggregateAgentTrustMetric:
    """Get metrics for the trust scores on agents"""

    # assign agent tracks to agent viewable truths
    agent_metrics = {}
    timestamp = 0.0
    for ID_agent in truths_agents:
        timestamp = tracks_agents[ID_agent].timestamp
        A_truth = build_A_from_distance(
            tracks_agents[ID_agent],
            truths_agents[ID_agent],
            check_reference=False,
        )
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
            timestamp=timestamp,
            identifier=ID_agent,
            f1_score=confusion.f1_score,
            area_above_cdf=area_above_cdf(trust_agents[ID_agent]),
            f1_threshold=f1_threshold,
            agent_is_attacked=ID_agent in attacked_agents,
        )

    return AggregateAgentTrustMetric(timestamp=timestamp, agent_metrics=agent_metrics)


def get_trust_tracks_metrics(
    truths: "DataContainer",
    tracks_cc: "DataContainer",
    trust_tracks: "TrustArray",
    assign_radius: float = 2.0,
) -> AggregateTrackTrustMetric:
    """Get metrics for the trust scores on tracks"""

    # assign cc tracks to truths
    A = build_A_from_distance(
        tracks_cc,
        truths,
        check_reference=False,
    )
    assigns = gnn_single_frame_assign(A, cost_threshold=assign_radius)
    IDs_assign = [tracks_cc[j_track].ID for j_track in assigns._row_to_col]

    # get the metric for each track
    track_metrics = {}
    timestamp = tracks_cc.timestamp
    for track in tracks_cc:
        track_metrics[track.ID] = TrackTrustMetric(
            identifier=track.ID,
            timestamp=timestamp,
            area_above_cdf=area_above_cdf(trust_tracks[track.ID]),
            assigned_to_truth=(track.ID in IDs_assign),
        )

    return AggregateTrackTrustMetric(timestamp=timestamp, track_metrics=track_metrics)
