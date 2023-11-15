from avstack.config import AGENTS, MODELS, Config

import mate


cfg_w = Config.fromfile("config/_base_/base_world.py")
world = MODELS.build(cfg_w.world)


def test_build_agent():
    cfg_a = Config.fromfile("config/_base_/base_agent.py")
    agent = AGENTS.build(cfg_a.agent, default_args={"world": world})
    assert isinstance(agent, mate.agents.Agent)
    assert isinstance(agent.pipeline, mate.pipeline.AgentPipeline)
