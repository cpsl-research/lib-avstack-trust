from typing import List

from avstack.config import ConfigDict

from .config import MATE
from .distributions import TrustArray, TrustBetaDistribution
from .measurement import PsmArray


class TrustUpdater:
    def __init__(
        self,
        prior_agents={},
        prior_tracks={},
        agent_propagator: ConfigDict = {
            "type": "PriorInterpolationPropagator",
            "prior": {
                "type": "TrustBetaDistribution",
                "timestamp": 0.0,
                "identifier": "prior",
                "alpha": 0.5,
                "beta": 0.5,
            },
            "dt_return": 10,
        },
        track_propagator: ConfigDict = {
            "type": "PriorInterpolationPropagator",
            "prior": {
                "type": "TrustBetaDistribution",
                "timestamp": 0.0,
                "identifier": "prior",
                "alpha": 0.5,
                "beta": 0.5,
            },
            "dt_return": 10,
        },
    ):
        self.agent_propagator = MATE.build(agent_propagator)
        self.track_propagator = MATE.build(track_propagator)
        self._prior_agents = prior_agents
        self._prior_tracks = prior_tracks
        self._prior_means = {"distrusted": 0.2, "untrusted": 0.5, "trusted": 0.8}
        self._set_data_structures()

    def _set_data_structures(self):
        self.trust_agents = TrustArray(timestamp=0.0, trusts=[])
        self.trust_tracks = TrustArray(timestamp=0.0, trusts=[])
        self._inactive_track_trust = TrustArray(timestamp=0.0, trusts=[])

    def reset(self):
        self._set_data_structures()

    # ==========================================
    # initialization
    # ==========================================

    def init_trust_distribution(self, timestamp: float, identifier: str, prior: dict):
        mean = self._prior_means[prior["type"]]
        precision = prior["strength"]
        alpha = mean * precision
        beta = (1 - mean) * precision
        return TrustBetaDistribution(
            timestamp=timestamp, identifier=identifier, alpha=alpha, beta=beta
        )

    def init_new_agents(self, timestamp: float, agent_ids: List[int]):
        # check for new agents
        for i_agent in agent_ids:
            if i_agent not in self.trust_agents:
                prior = self._prior_agents.get(
                    i_agent, {"type": "untrusted", "strength": 1}
                )
                self.trust_agents.append(
                    self.init_trust_distribution(timestamp, i_agent, prior)
                )

        # check for old agents

    def init_new_tracks(self, timestamp: float, track_ids: List[int]):
        # check for new tracks
        for track_id in track_ids:
            if track_id not in self.trust_tracks:
                prior = self._prior_tracks.get(
                    track_id, {"type": "untrusted", "strength": 1}
                )
                self.trust_tracks.append(
                    self.init_trust_distribution(timestamp, track_id, prior)
                )

        # check for old tracks

    # ==========================================
    # propagators
    # ==========================================

    def propagate_agent_trust(self, timestamp: float):
        """Propagate agent trust distributions"""
        self.trust_agents.propagate(timestamp, self.agent_propagator)

    def propagate_track_trust(self, timestamp: float):
        """Propagate track trust distributions"""
        self.trust_tracks.propagate(timestamp, self.track_propagator)

    # ==========================================
    # updaters
    # ==========================================

    def update_agent_trust(self, psms_agents: PsmArray):
        # self.propagate_agent_trust(psms_agents.timestamp)
        psms_agents_target = psms_agents.reduce_by_target()
        for i_agent, psms in psms_agents_target.items():
            for psm in psms:
                self.trust_agents[i_agent].update(psm)
        return self.trust_agents

    def update_track_trust(self, psms_tracks: PsmArray):
        # self.propagate_track_trust(psms_tracks.timestamp)
        psms_tracks_target = psms_tracks.reduce_by_target()
        for ID_track, psms in psms_tracks_target.items():
            # ***enforce constraint on number of updates***
            # if a track has only a single positive PSM, we cannot use it to
            # update, otherwise we fall into an echo-chamber effect where an
            # agent will continue to increase the trust score even if
            # no other agent can verify the existence
            if len(psms) > 1:
                for psm in psms:
                    self.trust_tracks[ID_track].update(psm)
        return self.trust_tracks