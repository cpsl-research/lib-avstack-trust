from avstack.config import AGENTS, MODELS, Config

import mate


cfg_w = Config.fromfile("config/_base_/base_world.py")
world = MODELS.build(cfg_w.world)


def test_build_base_command_center():
    cfg_c = Config.fromfile("config/_base_/base_command_center.py")
    cc = AGENTS.build(cfg_c.commandcenter, default_args={"world": world})
    assert isinstance(cc, mate.agents.CommandCenter)
    assert isinstance(cc.pipeline, mate.pipeline.CommandCenterPipeline)


def test_build_trust_command_center():
    cfg_c = Config.fromfile("config/command_center/point_based_ci_fusion.py")
    cc = AGENTS.build(cfg_c.commandcenter, default_args={"world": world})
    assert isinstance(cc, mate.agents.CommandCenter)
    assert isinstance(cc.pipeline, mate.pipeline.CommandCenterPipeline)
