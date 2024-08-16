import logging
from typing import TYPE_CHECKING, Dict, List, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position, Shape
    from avstack.modules.clustering import Cluster
    from avstack.modules.tracking import TrackBase

import numpy as np
from avstack.config import MODELS, ConfigDict
from avstack.geometry.fov import points_in_fov
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .config import MATE
from .distributions import TrustDistribution


class Psm:
    def __init__(
        self,
        timestamp: float,
        target: str,
        value: float,
        confidence: float,
        source: str,
    ):
        self.timestamp = timestamp
        self.target = target
        self.value = value
        self.confidence = confidence
        self.source = source

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Pseudomeasurement: ({self.value}, {self.confidence})"


class PsmArray:
    def __init__(self, timestamp: float, psms: List[Psm]):
        self.timestamp = timestamp
        self.psms = psms

    def __add__(self, other: "PsmArray"):
        return PsmArray(self.timestamp, self.psms + other.psms)

    def __iter__(self):
        return iter(self.psms)

    def __getitem__(self, key: int) -> Psm:
        return self.psms[key]
    
    def __len__(self) -> int:
        return len(self.psms)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"PsmArray at time {self.timestamp}, {self.psms}"
    
    def append(self, other: "Psm"):
        self.psms.append(other)

    def extend(self, other: "PsmArray"):
        self.psms.extend(other.psms)

    def reduce_by_target(self) -> Dict[int, "PsmArray"]:
        psms_target = {}
        for psm in self.psms:
            if psm.target not in psms_target:
                psms_target[psm.target] = PsmArray(self.timestamp, [])
            psms_target[psm.target].append(psm)
        return psms_target


class PsmGenerator:
    def psms_agents(self):
        raise NotImplementedError

    def psms_agent(self):
        raise NotImplementedError

    def psms_tracks(self):
        raise NotImplementedError

    def psms_track(self):
        raise NotImplementedError


@MATE.register_module()
class ViewBasedPsm(PsmGenerator):
    """Simple overlap-based PSM function from CDC submission"""

    def __init__(
        self,
        assign_radius: float = 1.0,
        min_prec: float = 0.0,
        clusterer: ConfigDict = {
            "type": "SampledAssignmentClusterer",
            "assign_radius": 1.5,
        },
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.assign_radius = assign_radius
        self.min_prec = min_prec
        self.clusterer = MODELS.build(clusterer)
        self.verbose = verbose

    def psms_agents(
        self,
        fov_agents: Dict[int, "Shape"],
        tracks_agents: Dict[int, "DataContainer"],
        tracks_cc: "DataContainer",
        trust_tracks: Dict[int, TrustDistribution],
    ):
        """Obtains PSMs for all agents"""
        psms_agents = PsmArray(timestamp=tracks_cc.timestamp, psms=[])
        for i_agent in fov_agents:
            psms_agents.extend(
                self.psms_agent(
                    i_agent=i_agent,
                    fov_agent=fov_agents[i_agent],
                    tracks_agent=tracks_agents[i_agent],
                    tracks_cc=tracks_cc,
                    trust_tracks=trust_tracks,
                )
            )
        return psms_agents

    def psms_agent(
        self,
        i_agent: int,
        fov_agent: Union["Shape", np.ndarray],
        tracks_agent: "DataContainer",
        tracks_cc: "DataContainer",
        trust_tracks: Dict[int, TrustDistribution],
    ):
        """Creates PSMs for one agent"""
        # assign agent tracks to central tracks
        A = build_A_from_distance(
            [t.x[:2] for t in tracks_agent],
            [t.x[:2] for t in tracks_cc],
            check_reference=False,
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)
        psms = []

        # assignments provide psms proportional to track trust
        for _, j_trk_central in assign.iterate_over("rows", with_cost=False).items():
            j_trk_central = j_trk_central[0]  # assumes one assignment
            ID_central = tracks_cc[j_trk_central].ID
            if trust_tracks[ID_central].precision > self.min_prec:
                value = trust_tracks[ID_central].mean
                confidence = 1 - trust_tracks[ID_central].variance
                psms.append(
                    Psm(
                        timestamp=tracks_cc.timestamp,
                        target=i_agent,
                        value=value,
                        confidence=confidence,
                        source=ID_central,
                    )
                )
            else:
                raise NotImplementedError()

        # tracks in local not in central are unclear...
        for i_trk_agent in assign.unassigned_rows:
            pass

        # tracks in central not in local provide psms inversely proportional to trust
        for j_trk_central in assign.unassigned_cols:
            if points_in_fov(tracks_cc[j_trk_central].x[:2], fov_agent):
                ID_central = tracks_cc[j_trk_central].ID
                if trust_tracks[ID_central].precision > self.min_prec:
                    value = 1 - trust_tracks[ID_central].mean
                    confidence = 1 - trust_tracks[ID_central].variance
                    psms.append(
                        Psm(
                            timestamp=tracks_cc.timestamp,
                            target=i_agent,
                            value=value,
                            confidence=confidence,
                            source=ID_central,
                        )
                    )
                else:
                    raise NotImplementedError()

        return PsmArray(timestamp=tracks_cc.timestamp, psms=psms)

    def psms_tracks(
        self,
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, "Shape"],
        tracks_agents: Dict[int, "DataContainer"],
        tracks_cc: "DataContainer",
        trust_agents: Dict[int, TrustDistribution],
    ):
        """Obtains PSMs for all tracks"""
        psms_tracks = PsmArray(timestamp=tracks_cc.timestamp, psms=[])

        # assign clusters to existing tracks for IDs
        clusters = self.clusterer(tracks_agents, frame=0, timestamp=0)
        A = build_A_from_distance(
            [c.centroid()[:2] for c in clusters], [t.x[:2] for t in tracks_cc]
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)

        # assignments - run pseudomeasurement generation
        for j_clust, i_track in assign.iterate_over("rows", with_cost=False).items():
            i_track = i_track[0]  # one edge only
            psms_tracks.extend(
                self.psms_track_assigned(
                    position_agents=position_agents,
                    fov_agents=fov_agents,
                    trust_agents=trust_agents,
                    cluster=clusters[j_clust],
                    track_cc=tracks_cc[i_track],
                    timestamp=tracks_cc.timestamp,
                )
            )

        # lone clusters - do not do anything, assume they start new tracks
        for j_clust in assign.unassigned_rows:
            pass

        # lone tracks - penalize because of no detections (if in view)
        for i_track in assign.unassigned_cols:
            psms_tracks.extend(
                self.psms_track_unassigned(
                    position_agents=position_agents,
                    fov_agents=fov_agents,
                    trust_agents=trust_agents,
                    track_cc=tracks_cc[i_track],
                    timestamp=tracks_cc.timestamp,
                )
            )

        return psms_tracks

    def psms_track_assigned(
        self,
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, Union["Shape", np.ndarray]],
        trust_agents: Dict[int, TrustDistribution],
        cluster: "Cluster",
        track_cc: "TrackBase",
        timestamp: float,
        d_thresh_self: float = 1.0,
    ):
        """Creates PSMs for one track"""
        psms = []
        for i_agent in fov_agents:
            if points_in_fov(track_cc.x[:2], fov_agents[i_agent]):
                # Get the PSM values
                saw = i_agent in cluster.agent_IDs  # did this agent see it?
                if saw:  # positive result
                    dets = cluster.get_objects_by_agent_ID(i_agent)
                    if len(dets) > 1:
                        if self.verbose:
                            logging.warning(
                                "Clustered more than one detection to track..."
                            )
                    value = 1.0
                    confidence = trust_agents[i_agent].mean
                else:  # negative result
                    # Handle case where agent can't see itself
                    if (
                        np.linalg.norm(position_agents[i_agent].x[:3] - track_cc.x[:3])
                        < d_thresh_self
                    ):
                        value = 1.0
                        confidence = 1.0
                    else:
                        value = 0.0
                        confidence = trust_agents[i_agent].mean
                # Construct the psm
                psm = Psm(
                    timestamp=timestamp,
                    target=track_cc.ID,
                    value=value,
                    confidence=confidence,
                    source=i_agent,
                )
                psms.append(psm)
        return PsmArray(timestamp=timestamp, psms=psms)

    def psms_track_unassigned(
        self,
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, Union["Shape", np.ndarray]],
        trust_agents: Dict[int, TrustDistribution],
        track_cc: "TrackBase",
        timestamp: float,
    ):
        psms = []
        for i_agent in fov_agents:
            if points_in_fov(track_cc.x[:2], fov_agents[i_agent]):
                psm = Psm(
                    timestamp=timestamp,
                    target=track_cc.ID,
                    value=0.0,
                    confidence=trust_agents[i_agent].mean,
                    source=i_agent,
                )
                psms.append(psm)
        return PsmArray(timestamp=timestamp, psms=psms)
