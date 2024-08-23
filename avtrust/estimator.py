from typing import TYPE_CHECKING, Dict, Tuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position, Shape

    from .measurement import PsmArray, PsmGenerator
    from .updater import TrustArray, TrustUpdater


class TrustEstimator:
    def __init__(self, measurement: "PsmGenerator", updater: "TrustUpdater"):
        self.measurement = measurement
        self.updater = updater

    def __call__(
        self,
        position_agents: Dict[str, "Position"],
        fov_agents: Dict[str, "Shape"],
        tracks_agents: Dict[str, "DataContainer"],
        tracks_cc: "DataContainer",
    ) -> Tuple["TrustArray", "TrustArray", "PsmArray", "PsmArray"]:
        # Init new distributions if needed
        timestamp = tracks_cc.timestamp
        self.updater.init_new_agents(timestamp, list(position_agents.keys()))
        self.updater.init_new_tracks(timestamp, [track.ID for track in tracks_cc])

        # Generate PSM measurements
        psms_agents, psms_tracks = self.measurement(
            position_agents=position_agents,
            fov_agents=fov_agents,
            tracks_agents=tracks_agents,
            tracks_cc=tracks_cc,
            trust_agents=self.updater.trust_agents,
            trust_tracks=self.updater.trust_tracks,
        )

        # Update trust with measurements
        trust_agents = self.updater.update_agent_trust(psms_agents)
        trust_tracks = self.updater.update_track_trust(psms_tracks)

        return trust_agents, trust_tracks, psms_agents, psms_tracks

    def reset(self):
        self.updater.reset()

    def init_new_agents(self, timestamp: float, agent_ids):
        self.updater.init_new_agents(timestamp, agent_ids)

    def init_new_tracks(self, timestamp: float, track_ids):
        self.updater.init_new_tracks(timestamp, track_ids)
