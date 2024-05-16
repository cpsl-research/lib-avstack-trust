import logging
from typing import TYPE_CHECKING, Dict, Union

import numpy as np


if TYPE_CHECKING:
    from avstack.geometry.fov import Shape
    from avstack.datastructs import DataContainer
    from avstack.modules.clustering import Cluster
    from avstack.modules.tracking.tracks import _TrackBase

from avstack.config import MODELS, ConfigDict
from avstack.geometry.fov import points_in_fov
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign

from .distributions import TrustBetaParams
from .psm import PSM


class TrustEstimator:
    def __init__(
        self,
        Pd=0.9,
        delta_m_factor=50,
        delta_v_factor=0.05,
        do_propagate=True,
        prior_agents={},
        prior_tracks={},
        propagation_model="uncertainty",
        clusterer: ConfigDict = {
            "type": "SampledAssignmentClusterer",
            "assign_radius": 1.0,
        },
        assign_threshold: float = 1.0,
    ):
        self.Pd = Pd
        self.clusterer = MODELS.build(clusterer)
        self.delta_m_factor = delta_m_factor
        self.delta_v_factor = delta_v_factor
        self.do_propagate = do_propagate
        self._propagation_model = propagation_model
        self._prior_agents = prior_agents
        self._prior_tracks = prior_tracks
        self._prior_means = {"distrusted": 0.2, "untrusted": 0.5, "trusted": 0.8}
        self.tracks = []
        self.agent_trust = {}
        self.track_trust = {}
        self._inactive_track_trust = {}
        self.assign_threshold = assign_threshold

    def __call__(
        self,
        agents: Dict[int, np.ndarray],
        fovs: Dict[int, Union["Shape", np.ndarray]],
        dets: Dict[int, "DataContainer"],
        tracks: Dict[int, "DataContainer"],
        agent_tracks: Dict[int, "DataContainer"],
    ):
        if self.do_propagate:
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
        self.update_agent_trust(fovs, agent_tracks)

        # -- update object trust
        clusters = self.update_track_trust(agents, fovs, dets, tracks)

        return clusters

    def init_trust_distribution(self, prior):
        mean = self._prior_means[prior["type"]]
        precision = prior["strength"]
        alpha = mean * precision
        beta = (1 - mean) * precision
        return TrustBetaParams(alpha, beta)

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

    def psm_track(
        self,
        agents: Dict[int, np.ndarray],
        fovs: Dict[int, Union["Shape", np.ndarray]],
        cluster: "Cluster",
        track: "_TrackBase",
    ):
        psms = []
        for i_agent in fovs:
            if points_in_fov(cluster.centroid().x[:2], fovs[i_agent]):
                # Get the PSM
                saw = i_agent in cluster.agent_IDs  # did we see it?
                if saw:  # positive result
                    dets = cluster.get_objects_by_agent_ID(i_agent)
                    if len(dets) > 1:
                        logging.warning("Clustered more than one detection to track...")
                    psm = PSM(value=1.0, confidence=self.agent_trust[i_agent].mean)
                else:  # negative result
                    psm = PSM(value=0.0, confidence=self.agent_trust[i_agent].mean)
                psms.append(psm)
            else:
                pass  # not expected to see
        return psms

    def psm_agent(
        self,
        fov: Union["Shape", np.ndarray],
        tracks_agent: "DataContainer",
        tracks_central: "DataContainer",
        min_prec=0,
    ):
        # assign agent tracks to central tracks
        A = build_A_from_distance(
            [t.x[:2] for t in tracks_agent],
            [t.x[:2] for t in tracks_central],
            check_reference=False,
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_threshold)

        # assignments provide psms proportional to track trust
        means = []
        confidences = []
        for i_trk_agent, j_trk_central in assign.iterate_over(
            "rows", with_cost=False
        ).items():
            j_trk_central = j_trk_central[0]  # assumes one assignment
            ID_central = tracks_central[j_trk_central].ID
            if self.track_trust[ID_central].precision > min_prec:
                means.append(self.track_trust[ID_central].mean)
                confidences.append(1 - self.track_trust[ID_central].variance)

        # tracks in central not in local provide psms inversely proportional to trust
        for j_trk_central in assign.unassigned_cols:
            if points_in_fov(tracks_central[j_trk_central].x[:2], fov):
                ID_central = tracks_central[j_trk_central].ID
                if self.track_trust[ID_central].precision > min_prec:
                    mean = self.track_trust[ID_central].mean
                    confidence = 1 - self.track_trust[ID_central].variance
                    means.append(mean)
                    confidences.append(confidence)

        # tracks in local not in central are unclear...
        # TODO

        # reduce to a single PSM per frame
        # TODO: this is not a good way to do it
        psms = [PSM(value=np.mean(means), confidence=np.mean(confidences))]
        return psms

    def propagate(self, w_prior=0.2):
        # Each frame we add some uncertainty
        for ID in self.track_trust:
            if self._propagation_model == "uncertainty":
                # add variance
                a = self.track_trust[ID].alpha
                b = self.track_trust[ID].beta
                m = a / (a + b)
                v = a + b
                m += (0.5 - m) / self.delta_m_factor
                v += self.delta_v_factor
                self.track_trust[ID].alpha = m * v
                self.track_trust[ID].beta = (1 - m) * v
            elif self._propagation_model == "prior":
                w = w_prior
                prior = self.init_trust_distribution(
                    self._prior_tracks.get(ID, {"type": "untrusted", "strength": 1})
                )
                self.track_trust[ID].alpha = (1 - w) * self.track_trust[
                    ID
                ].alpha + w * prior.alpha
                self.track_trust[ID].beta = (1 - w) * self.track_trust[
                    ID
                ].beta + w * prior.beta
            elif self._propagation_model == "normalize":
                # perform a heuristic normalization on params
                s = (self.track_trust[ID].alpha + self.track_trust[ID].beta) / 2
                self.track_trust[ID].alpha /= s
                self.track_trust[ID].beta /= s
            else:
                raise NotImplementedError

    def update_track_trust(self, agents, fovs, dets, tracks):
        # cluster the detections
        clusters = self.clusterer(dets, frame=0, timestamp=0)

        # assign clusters to existing tracks for IDs
        A = build_A_from_distance(clusters, tracks)
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_threshold)

        # assignments - run pseudomeasurement generation
        for j_clust, i_track in assign.iterate_over("rows", with_cost=False).items():
            i_track = i_track[0]  # one edge only
            ID_track = tracks[i_track].ID
            psms = self.psm_track(agents, fovs, clusters[j_clust], tracks[i_track])

            # update the parameters
            if len(psms) > 1:
                for psm in psms:
                    self.track_trust[ID_track].alpha += psm.confidence * psm.value
                    self.track_trust[ID_track].beta += psm.confidence * (1 - psm.value)

        # lone clusters - do not do anything, assume they start new tracks
        if len(assign.unassigned_rows) > 0:
            pass

        # lone tracks - penalize because of no detections (if in view)
        if len(assign.unassigned_cols) > 0:
            for i_track in assign.unassigned_cols:
                ID_track = tracks[i_track].ID
                psm = PSM(
                    value=0.0, confidence=1.0
                )  # TODO: merge this in with other PSM generation
                self.track_trust[ID_track].alpha += psm.confidence * psm.value
                self.track_trust[ID_track].beta += psm.confidence * (1 - psm.value)

        return clusters

    def update_agent_trust(self, fovs, agent_tracks):
        for i_agent in fovs:
            psms = self.psm_agent(fovs[i_agent], agent_tracks[i_agent], self.tracks)
            for psm in psms:
                self.agent_trust[i_agent].alpha += psm.confidence * psm.value
                self.agent_trust[i_agent].beta += psm.confidence * (1 - psm.value)
