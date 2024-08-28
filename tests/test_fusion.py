from typing import NamedTuple

from avstack.datastructs import DataContainer

from avtrust.fusion import TrackThresholdingTrustFusion


class FakeTrack(NamedTuple):
    ID: int


class FakeTrust(NamedTuple):
    mean: float


def test_track_thresholding_fusion():
    n_tracks = 10
    trust_tracks = {
        i: FakeTrust(0.2) if i % 2 else FakeTrust(0.8) for i in range(n_tracks)
    }
    threshold = 0.40
    tracks_cc = DataContainer(
        frame=0,
        timestamp=0,
        source_identifier="tracks",
        data=[FakeTrack(i) for i in range(n_tracks)],
    )
    fusion = TrackThresholdingTrustFusion(threshold_track_ignore=threshold)
    tracks_out = fusion(trust_tracks=trust_tracks, tracks_cc=tracks_cc)
    assert len(tracks_out) == sum([v.mean >= threshold for v in trust_tracks.values()])
