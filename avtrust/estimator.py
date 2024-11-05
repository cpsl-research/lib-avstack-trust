from typing import TYPE_CHECKING, Dict, Tuple


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position, Shape

    from .measurement import PsmArray, PsmGenerator
    from .updater import TrustArray, TrustUpdater


from avtrust.config import AVTRUST


class TrustEstimator:
    def __init__(
        self,
        measurement: "PsmGenerator",
        updater: "TrustUpdater",
    ):
        self.measurement = (
            AVTRUST.build(measurement) if isinstance(measurement, dict) else measurement
        )
        self.updater = AVTRUST.build(updater) if isinstance(updater, dict) else updater

    @property
    def trust_agents(self):
        return self.updater.trust_agents

    @property
    def trust_tracks(self):
        return self.updater.trust_tracks

    def reset(self):
        self.updater.reset()

    def init_new_agents(self, timestamp: float, agent_ids):
        self.updater.init_new_agents(timestamp, agent_ids)

    def init_new_tracks(self, timestamp: float, track_ids):
        self.updater.init_new_tracks(timestamp, track_ids)


@AVTRUST.register_module()
class CentralizedTrustEstimator(TrustEstimator):
    def __call__(
        self,
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, "Shape"],
        tracks_agents: Dict[int, "DataContainer"],
        tracks_cc: "DataContainer",
    ) -> Tuple["TrustArray", "TrustArray", "PsmArray", "PsmArray"]:
        """Handles centralized trust updating jointly"""

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


# @AVTRUST.register_module()
# class DistributedTrustEstimator(TrustEstimator):
#     def __call__(
#         self,
#         position_ego: "Position",
#         fov_ego: "Shape",
#         tracks_ego: "DataContainer",
#         id_agent_received: int,
#         position_received: "Position",
#         tracks_received: "DataContainer",
#         fov_received: "Shape",
#     ) -> Tuple["TrustArray", "TrustArray", "PsmArray", "PsmArray"]:
#         """Handles one at a time trust updating vs ego data"""

#         # Init new distributions if needed
#         timestamp = tracks_ego.timestamp
#         self.updater.init_new_agents(timestamp, [id_agent_received])
#         self.updater.init_new_tracks(timestamp, [track.ID for track in tracks_ego])

#         # Generate PSM measurements
#         psms_agents, psms_tracks = self.measurement(
#             position_agents=position_agents,
#             fov_agents=fov_agents,
#             tracks_agents=tracks_agents,
#             tracks_cc=tracks_cc,
#             trust_agents=self.updater.trust_agents,
#             trust_tracks=self.updater.trust_tracks,
#         )

#         # Update trust with measurements
#         trust_agents = self.updater.update_agent_trust(psms_agents)
#         trust_tracks = self.updater.update_track_trust(psms_tracks)

#         return trust_agents, trust_tracks, psms_agents, psms_tracks
