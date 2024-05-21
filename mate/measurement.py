import logging
from typing import TYPE_CHECKING, Dict, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Shape
    from avstack.modules.clustering import Cluster

import numpy as np
from avstack.geometry.fov import points_in_fov
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .config import MATE
from .distributions import TrustDistribution


class PSM:
    def __init__(self, value, confidence):
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Pseudomeasurement: ({self.value}, {self.confidence})"


class PsmGenerator:
    def psm_agents(self):
        raise NotImplementedError

    def psm_agent(self):
        raise NotImplementedError

    def psm_tracks(self):
        raise NotImplementedError

    def psm_track(self):
        raise NotImplementedError


@MATE.register_module()
class ViewBasedPsm(PsmGenerator):
    """Simple overlap-based PSM function from CDC submission"""

    def __init__(self, assign_radius: float = 1.0, min_prec: float = 0.0) -> None:
        super().__init__()
        self.assign_radius = assign_radius
        self.min_prec = min_prec

    def psm_agents(self, fovs, tracks_agents, tracks, track_trust):
        psms_agents = {i_agent: [] for i_agent in fovs}  # this is unnecessary
        for i_agent in fovs:
            psms_agents[i_agent] = self.psm_agent(
                fov=fovs[i_agent],
                tracks_agent=tracks_agents[i_agent],
                tracks=tracks,
                track_trust=track_trust,
            )
        return psms_agents

    def psm_agent(
        self,
        fov: Union["Shape", np.ndarray],
        tracks_agent: "DataContainer",
        tracks: "DataContainer",
        track_trust: Dict[int, TrustDistribution],
    ):
        """Creates PSM for one agent"""
        # assign agent tracks to central tracks
        A = build_A_from_distance(
            [t.x[:2] for t in tracks_agent],
            [t.x[:2] for t in tracks],
            check_reference=False,
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)

        # assignments provide psms proportional to track trust
        means = []
        confidences = []
        for i_trk_agent, j_trk_central in assign.iterate_over(
            "rows", with_cost=False
        ).items():
            j_trk_central = j_trk_central[0]  # assumes one assignment
            ID_central = tracks[j_trk_central].ID
            if track_trust[ID_central].precision > self.min_prec:
                means.append(track_trust[ID_central].mean)
                confidences.append(1 - track_trust[ID_central].variance)

        # tracks in central not in local provide psms inversely proportional to trust
        for j_trk_central in assign.unassigned_cols:
            if points_in_fov(tracks[j_trk_central].x[:2], fov):
                ID_central = tracks[j_trk_central].ID
                if track_trust[ID_central].precision > self.min_prec:
                    mean = track_trust[ID_central].mean
                    confidence = 1 - track_trust[ID_central].variance
                    means.append(mean)
                    confidences.append(confidence)

        # tracks in local not in central are unclear...
        # TODO

        # reduce to a single PSM per frame
        # TODO: this is not a good way to do it
        psms = [PSM(value=np.mean(means), confidence=np.mean(confidences))]
        return psms

    def psm_tracks(self, agents, fovs, agent_trust, clusters, tracks, assign):
        psms_tracks = {i_track: [] for i_track in tracks}  # this is unnecessary

        # assignments - run pseudomeasurement generation
        for j_clust, i_track in assign.iterate_over("rows", with_cost=False).items():
            i_track = i_track[0]  # one edge only
            ID_track = tracks[i_track].ID
            psms_tracks[ID_track] = self.psm_track(
                agents, fovs, agent_trust, clusters[j_clust]
            )

        # lone clusters - do not do anything, assume they start new tracks
        for j_clust in assign.unassigned_rows:
            pass

        # lone tracks - penalize because of no detections (if in view)
        for i_track in assign.unassigned_cols:
            ID_track = tracks[i_track].ID
            psms_tracks[ID_track] = [PSM(value=0.0, confidence=1.0)]
        return psms_tracks

    def psm_track(
        self,
        agents: Dict[int, np.ndarray],
        fovs: Dict[int, Union["Shape", np.ndarray]],
        agent_trusts: Dict[int, TrustDistribution],
        cluster: "Cluster",
    ):
        """Creates PSM for one cluster"""
        psms = []
        for i_agent in fovs:
            if points_in_fov(cluster.centroid().x[:2], fovs[i_agent]):
                # Get the PSM
                saw = i_agent in cluster.agent_IDs  # did this agent see it?
                if saw:  # positive result
                    dets = cluster.get_objects_by_agent_ID(i_agent)
                    if len(dets) > 1:
                        logging.warning("Clustered more than one detection to track...")
                    psm = PSM(value=1.0, confidence=agent_trusts[i_agent].mean)
                else:  # negative result
                    psm = PSM(value=0.0, confidence=agent_trusts[i_agent].mean)
                psms.append(psm)
            else:
                pass  # not expected to see
        return psms
