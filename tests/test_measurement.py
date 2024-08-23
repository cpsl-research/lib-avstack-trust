import numpy as np
from avstack.datastructs import DataContainer
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
from avstack.geometry.fov import Circle
from avstack.modules.tracking import BasicBoxTrack3D

from avtrust.distributions import TrustArray, TrustBetaDistribution
from avtrust.measurement import ViewBasedPsm


def make_one_agent_cc_track(ID_agent=0, ID_track=10, dy_offset=0.0):
    position_agents = {
        ID_agent: Position(x=np.array([20, 20, 20]), reference=GlobalOrigin3D)
    }
    fov_agents = {ID_agent: Circle(radius=5, center=np.array([0, 0]))}
    box_track_1 = BasicBoxTrack3D(
        t0=0.0,
        box3d=Box3D(
            position=Position(np.array([0.5, 0, 0]), reference=GlobalOrigin3D),
            attitude=Attitude(q=np.quaternion(1), reference=GlobalOrigin3D),
            hwl=[2, 2, 2],
        ),
        ID_force=ID_track,
        obj_type="car",
        reference=GlobalOrigin3D,
    )
    box_track_2 = BasicBoxTrack3D(
        t0=0.0,
        box3d=Box3D(
            position=Position(np.array([0.5, dy_offset, 0]), reference=GlobalOrigin3D),
            attitude=Attitude(q=np.quaternion(1), reference=GlobalOrigin3D),
            hwl=[2, 2, 2],
        ),
        ID_force=ID_track,
        obj_type="car",
        reference=GlobalOrigin3D,
    )
    tracks_agent = {
        ID_agent: DataContainer(
            timestamp=1.0, frame=0, source_identifier="tracks", data=[box_track_1]
        )
    }
    tracks_cc = DataContainer(
        timestamp=1.0,
        frame=0,
        source_identifier="tracks_cc",
        data=[box_track_2],
    )
    ag_trust = TrustBetaDistribution(timestamp=0.0, identifier=0, alpha=9, beta=1)
    tr_trust = TrustBetaDistribution(
        timestamp=0.0, identifier=ID_track, alpha=5, beta=1
    )
    trust_agents = TrustArray(timestamp=0.0, trusts=[ag_trust])
    trust_tracks = TrustArray(timestamp=0.0, trusts=[tr_trust])
    return (
        position_agents,
        fov_agents,
        tracks_agent,
        tracks_cc,
        trust_agents,
        trust_tracks,
    )


def test_psm_one_agent_one_track_close():
    psm_generator = ViewBasedPsm(assign_radius=1.0)
    ID_agent = 0
    ID_track = 10
    (
        position_agents,
        fov_agents,
        tracks_agent,
        tracks_cc,
        trust_agents,
        trust_tracks,
    ) = make_one_agent_cc_track(ID_agent=ID_agent, ID_track=ID_track, dy_offset=0.25)
    psms_agents, psms_tracks = psm_generator(
        position_agents=position_agents,
        fov_agents=fov_agents,
        tracks_agents=tracks_agent,
        tracks_cc=tracks_cc,
        trust_tracks=trust_tracks,
        trust_agents=trust_agents,
    )

    # check agent psms
    assert len(psms_agents) == 1
    assert psms_agents[0].value == trust_tracks[tracks_cc[0].ID].mean
    assert psms_agents[0].confidence == (1 - trust_tracks[tracks_cc[0].ID].variance)

    # check track psms
    assert len(psms_tracks) == 1
    assert psms_tracks[0].value == 1.0
    assert psms_tracks[0].confidence == trust_agents[ID_agent].mean


def test_psm_one_agent_one_track_far():
    psm_generator = ViewBasedPsm(assign_radius=1.0)
    ID_agent = 0
    ID_track = 10
    (
        position_agents,
        fov_agents,
        tracks_agent,
        tracks_cc,
        trust_agents,
        trust_tracks,
    ) = make_one_agent_cc_track(ID_agent=ID_agent, ID_track=ID_track, dy_offset=4.0)
    psms_agents, psms_tracks = psm_generator(
        position_agents=position_agents,
        fov_agents=fov_agents,
        tracks_agents=tracks_agent,
        tracks_cc=tracks_cc,
        trust_tracks=trust_tracks,
        trust_agents=trust_agents,
    )

    # check agent psms
    assert len(psms_agents) == 1
    assert psms_agents[0].value == 1 - trust_tracks[tracks_cc[0].ID].mean
    assert psms_agents[0].confidence == (1 - trust_tracks[tracks_cc[0].ID].variance)

    # check track psms
    assert len(psms_tracks) == 1
    assert psms_tracks[0].value == 0.0
    assert psms_tracks[0].confidence == trust_agents[ID_agent].mean
