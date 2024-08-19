from mate.updater import TrustUpdater


def test_init_new_agents():
    updater = TrustUpdater()
    assert len(updater.trust_agents) == 0
    updater.init_new_agents(timestamp=1.0, agent_ids=[0, 2, 4])
    assert len(updater.trust_agents) == 3


def test_init_new_tracks():
    updater = TrustUpdater()
    assert len(updater.trust_tracks) == 0
    updater.init_new_tracks(timestamp=1.0, track_ids=[1, 5, 6, 7])
    assert len(updater.trust_tracks) == 4
