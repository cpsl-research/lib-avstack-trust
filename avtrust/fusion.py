from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import Position, Shape
    from avstack.modules.tracking import TrackBase

    from .updater import TrustArray

from avtrust.config import AVTRUST


class TrustFusion:
    def __call__(
        self,
        trust_agents: "TrustArray",
        trust_tracks: "TrustArray",
        position_agents: Dict[int, "Position"],
        fov_agents: Dict[int, "Shape"],
        tracks_agents: Dict[int, "DataContainer"],
        tracks_cc: "DataContainer",
        *args,
        **kwargs,
    ) -> "DataContainer":
        """Takes as input all the information and outputs tracks

        Implemented in subclass
        """
        raise NotImplementedError


@AVTRUST.register_module()
class TrackThresholdingTrustFusion(TrustFusion):
    def __init__(self, threshold_track_ignore: float = 0.40):
        self.threshold_track_ignore = threshold_track_ignore

    def __call__(
        self,
        trust_tracks: "TrustArray",
        tracks_cc: "DataContainer",
        *args,
        **kwargs,
    ) -> "DataContainer":
        """Threshold any tracks that do not meet the trust criteria"""
        return tracks_cc.filter(self.track_is_valid, trust_tracks=trust_tracks)

    def track_is_valid(self, track: "TrackBase", trust_tracks: "TrustArray") -> bool:
        if track.ID in trust_tracks:
            return trust_tracks[track.ID].mean >= self.threshold_track_ignore
        else:
            return False
