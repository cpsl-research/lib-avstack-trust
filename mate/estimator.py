from typing import TYPE_CHECKING, Dict, Union

import numpy as np


if TYPE_CHECKING:
    from avstack.geometry.fov import Shape
    from avstack.datastructs import DataContainer

from avstack.config import MODELS, ConfigDict
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .config import MATE
from .distributions import TrustBetaDistribution


@MATE.register_module()
class TrustEstimator:
    def __init__(
        self,
        Pd=0.9,
        prior_agents={},
        prior_tracks={},
        agent_propagator: ConfigDict = {
            "type": "PriorInterpolationPropagator",
            "prior": {"type": "TrustBetaDistribution", "alpha": 0.5, "beta": 0.5},
            "w_prior": 0.1,
        },
        track_propagator: ConfigDict = {
            "type": "PriorInterpolationPropagator",
            "prior": {"type": "TrustBetaDistribution", "alpha": 0.5, "beta": 0.5},
            "w_prior": 0.1,
        },
        clusterer: ConfigDict = {
            "type": "SampledAssignmentClusterer",
            "assign_radius": 1.5,
        },
        psm: ConfigDict = {
            "type": "ViewBasedPsm",
            "assign_radius": 1.5,
        },
        assign_radius: float = 1.5,
    ):
        self.Pd = Pd
        self.clusterer = MODELS.build(clusterer)
        self.psm = MATE.build(psm)
        self.agent_propagator = MATE.build(agent_propagator)
        self.track_propagator = MATE.build(track_propagator)
        self._prior_agents = prior_agents
        self._prior_tracks = prior_tracks
        self._prior_means = {"distrusted": 0.2, "untrusted": 0.5, "trusted": 0.8}
        self.tracks = []
        self.agent_trust = {}
        self.track_trust = {}
        self._inactive_track_trust = {}
        self.assign_radius = assign_radius

    def __call__(
        self,
        agents: Dict[int, np.ndarray],
        fovs: Dict[int, Union["Shape", np.ndarray]],
        dets: Dict[int, "DataContainer"],
        tracks: Dict[int, "DataContainer"],
        agent_tracks: Dict[int, "DataContainer"],
    ):
        # -- run propagation
        self.propagate()

        # -- initialize new distributions
        self.tracks = tracks
        self.init_new_agents(agents, fovs)
        self.init_new_tracks(tracks)

        # -- prune old distributions
        track_IDs = [track.ID for track in tracks]
        # convert keys to list to handle popping during looping
        for track_trust_ID in list(self.track_trust.keys()):
            if track_trust_ID not in track_IDs:
                self._inactive_track_trust = self.track_trust[track_trust_ID]
                self.track_trust.pop(track_trust_ID)

        # -- update agent trust
        psms_agents = self.update_agent_trust(fovs, agent_tracks, tracks)

        # -- update object trust
        # psms_tracks, clusters = self.update_track_trust(agents, fovs, dets, tracks)
        psms_tracks, clusters = self.update_track_trust(
            agents, fovs, agent_tracks, tracks
        )

        return clusters, psms_agents, psms_tracks

    def init_trust_distribution(self, prior):
        mean = self._prior_means[prior["type"]]
        precision = prior["strength"]
        alpha = mean * precision
        beta = (1 - mean) * precision
        return TrustBetaDistribution(alpha, beta)

    def init_new_agents(self, agents, fovs):
        for i_agent in fovs:
            if i_agent not in self.agent_trust:
                prior = self._prior_agents.get(
                    i_agent, {"type": "untrusted", "strength": 1}
                )
                self.agent_trust[i_agent] = self.init_trust_distribution(prior)

    def init_new_tracks(self, tracks):
        for track in tracks:
            if track.ID not in self.track_trust:
                prior = self._prior_tracks.get(
                    track.ID, {"type": "untrusted", "strength": 1}
                )
                self.track_trust[track.ID] = self.init_trust_distribution(prior)

    def propagate(self):
        """Propagate agent and track trust distributions"""
        for ID in self.agent_trust:
            self.agent_propagator.propagate(self.agent_trust[ID])
        for ID in self.track_trust:
            self.track_propagator.propagate(self.track_trust[ID])

    def update_track_trust(self, agents, fovs, dets, tracks):
        # cluster the detections
        clusters = self.clusterer(dets, frame=0, timestamp=0)

        # assign clusters to existing tracks for IDs
        A = build_A_from_distance(
            [c.centroid()[:2] for c in clusters], [t.x[:2] for t in tracks]
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)

        # update the parameters from psms
        psms_tracks = self.psm.psm_tracks(
            agents, fovs, self.agent_trust, clusters, tracks, assign
        )
        for ID_track, psms in psms_tracks.items():
            for psm in psms:
                self.track_trust[ID_track].update(psm)

        return psms_tracks, clusters

    def update_agent_trust(self, fovs, tracks_agent, tracks):
        # update the parameters from psms
        psms_agents = self.psm.psm_agents(fovs, tracks_agent, tracks, self.track_trust)
        for i_agent, psms in psms_agents.items():
            for psm in psms:
                self.agent_trust[i_agent].update(psm)
        return psms_agents
