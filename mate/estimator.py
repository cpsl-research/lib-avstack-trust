from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .measurement import PsmGenerator
    from .updater import TrustUpdater


class TrustEstimator:
    def __init__(self, measurement: "PsmGenerator", updater: "TrustUpdater"):
        self.measurement = measurement
        self.updater = updater

    def __call__(self, position_agents, fov_agents, tracks_agents, tracks_cc):
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
