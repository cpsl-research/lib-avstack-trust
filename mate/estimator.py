import json
import os
from typing import TYPE_CHECKING, Dict, Union

import numpy as np


if TYPE_CHECKING:
    from avstack.geometry.fov import Shape
    from avstack.datastructs import DataContainer

from avstack.config import MODELS, ConfigDict
from avstack.modules import BaseModule
from avstack.modules.assignment import build_A_from_distance, gnn_single_frame_assign
from avstack.utils.decorators import apply_hooks

from .config import MATE
from .distributions import TrustBetaDistribution, TrustDistDecoder
from .utils import BetaDistWriter, PsmWriter


class TrustMessageEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, TrustMessage):
            trust_dict = {
                "agent_trust": {
                    ID: trust.encode() for ID, trust in o.agent_trust.items()
                },
                "track_trust": {
                    ID: trust.encode() for ID, trust in o.track_trust.items()
                },
                "frame": o.frame,
                "timestamp": o.timestamp,
            }
            return {"trust": trust_dict}
        else:
            raise NotImplementedError(f"{type(o)}, {o}")


class TrustMessageDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "trust" in json_object:
            json_object = json_object["trust"]
            return TrustMessage(
                frame=int(json_object["frame"]),
                timestamp=float(json_object["timestamp"]),
                agent_trust={
                    int(ID): json.loads(trust, cls=TrustDistDecoder)
                    for ID, trust in json_object["agent_trust"].items()
                },
                track_trust={
                    int(ID): json.loads(trust, cls=TrustDistDecoder)
                    for ID, trust in json_object["track_trust"].items()
                },
            )
        else:
            return json_object


class TrustMessage:
    def __init__(self, frame, timestamp, agent_trust, track_trust):
        self.frame = frame
        self.timestamp = timestamp
        self.agent_trust = agent_trust
        self.track_trust = track_trust

    def encode(self):
        return json.dumps(self, cls=TrustMessageEncoder)


@MATE.register_module()
class TrustEstimator(BaseModule):
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
        log_dir: str = "last_run",
        name: str = "trustestimator",
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        self.Pd = Pd
        self.clusterer = MODELS.build(clusterer)
        self.psm = MATE.build(psm)
        self.agent_propagator = MATE.build(agent_propagator)
        self.track_propagator = MATE.build(track_propagator)
        self._prior_agents = prior_agents
        self._prior_tracks = prior_tracks
        self._prior_means = {"distrusted": 0.2, "untrusted": 0.5, "trusted": 0.8}
        self.assign_radius = assign_radius
        self.log_dir = log_dir

        # set up data to track
        self._set_data_structures()
        self._set_log_file()

    def _set_data_structures(self):
        self.cc_tracks = []
        self.agent_trust = {}
        self.track_trust = {}
        self._inactive_track_trust = {}

    def _set_log_file(self):
        # -- set up logging
        os.makedirs(self.log_dir, exist_ok=True)
        # psm logging
        self.psm_agent_file = os.path.join(self.log_dir, "psm_agent_log.txt")
        self.psm_track_file = os.path.join(self.log_dir, "psm_track_log.txt")
        # trust dist logging
        self.trust_agent_file = os.path.join(self.log_dir, "trust_agent_log.txt")
        self.trust_track_file = os.path.join(self.log_dir, "trust_track_log.txt")
        # wipe the logs
        for file in [
            self.psm_agent_file,
            self.psm_track_file,
            self.trust_agent_file,
            self.trust_track_file,
        ]:
            open(file, "w").close()

    def reset(self):
        self._set_data_structures()
        self._set_log_file()

    @apply_hooks
    def __call__(
        self,
        frame: int,
        timestamp: float,
        agent_poses: Dict[int, np.ndarray],
        agent_fovs: Dict[int, Union["Shape", np.ndarray]],
        agent_tracks: Dict[int, "DataContainer"],
        cc_tracks: "DataContainer",
        *args,
        **kwargs,
    ):
        # -- run propagation
        self.propagate()

        # -- initialize new distributions
        self.cc_tracks = cc_tracks
        self.init_new_agents(agent_poses, agent_fovs)
        self.init_new_tracks(cc_tracks)

        # -- prune old distributions
        cc_track_IDs = [track.ID for track in cc_tracks]
        # convert keys to list to handle popping during looping
        for track_trust_ID in list(self.track_trust.keys()):
            if track_trust_ID not in cc_track_IDs:
                self._inactive_track_trust = self.track_trust[track_trust_ID]
                self.track_trust.pop(track_trust_ID)

        # -- update agent trust
        psms_agents = self.update_agent_trust(agent_fovs, agent_tracks, cc_tracks)

        # -- update object trust
        # psms_tracks, clusters = self.update_track_trust(agents, fovs, dets, tracks)
        psms_tracks, clusters = self.update_track_trust(
            agent_poses, agent_fovs, agent_tracks, cc_tracks
        )

        # log results
        PsmWriter.write(frame, psms_agents, self.psm_agent_file)
        PsmWriter.write(frame, psms_tracks, self.psm_track_file)
        BetaDistWriter.write(frame, self.agent_trust, self.trust_agent_file)
        BetaDistWriter.write(frame, self.track_trust, self.trust_track_file)

        return TrustMessage(
            frame=frame,
            timestamp=timestamp,
            agent_trust=self.agent_trust,
            track_trust=self.track_trust,
        )

    def init_trust_distribution(self, prior):
        mean = self._prior_means[prior["type"]]
        precision = prior["strength"]
        alpha = mean * precision
        beta = (1 - mean) * precision
        return TrustBetaDistribution(alpha, beta)

    def init_new_agents(self, agent_poses, fovs):
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

    def update_track_trust(self, agents, fovs, agent_tracks, cc_tracks):
        # cluster the detections
        clusters = self.clusterer(agent_tracks, frame=0, timestamp=0)

        # assign clusters to existing tracks for IDs
        A = build_A_from_distance(
            [c.centroid()[:2] for c in clusters], [t.x[:2] for t in cc_tracks]
        )
        assign = gnn_single_frame_assign(A, cost_threshold=self.assign_radius)

        # update the parameters from psms
        psms_tracks = self.psm.psm_tracks(
            agents, fovs, self.agent_trust, clusters, cc_tracks, assign
        )
        for ID_track, psms in psms_tracks.items():
            for psm in psms:
                self.track_trust[ID_track].update(psm)

        return psms_tracks, clusters

    def update_agent_trust(self, fovs, tracks_agent, cc_tracks):
        # update the parameters from psms
        psms_agents = self.psm.psm_agents(
            fovs, tracks_agent, cc_tracks, self.track_trust
        )
        for i_agent, psms in psms_agents.items():
            for psm in psms:
                self.agent_trust[i_agent].update(psm)
        return psms_agents
