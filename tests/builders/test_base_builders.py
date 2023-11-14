from avstack.config import AGENTS, MODELS, Config, ConfigDict

import mate


cfg_w = Config.fromfile("config/_base_/base_world.py")
world = MODELS.build(cfg_w.world)


def test_load_config():
    cfg = Config.fromfile("config/_base_/base_world.py")
    assert isinstance(cfg.world, ConfigDict)


def test_build_world():
    cfg = Config.fromfile("config/_base_/base_world.py")
    world = MODELS.build(cfg.world)
    assert isinstance(world, mate.simulation.world.World)


def test_build_object():
    cfg_o = Config.fromfile("config/_base_/base_object.py")
    obj = AGENTS.build(cfg_o.object, default_args={"world": world})
    assert isinstance(obj, mate.agents.Object)
