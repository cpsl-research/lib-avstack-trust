from avstack.config import AGENTS, MODELS, Config


def load_scenario_from_config_file(filename: str):
    cfg = Config.fromfile(filename=filename)
    return load_scenario_from_config(cfg)


def load_scenario_from_config(cfg):
    # -- world
    cfg_world = Config.fromfile(cfg.world).world
    world = MODELS.build(cfg_world)

    # -- objects
    objects = []
    for cfg_obj in cfg.objects:
        cfg_obj = Config.fromfile(cfg_obj).object
        obj = AGENTS.build(cfg_obj, default_args={"world": world})
        objects.append(obj)
        world.add_object(obj)
    
    # -- agents
    agents = []
    for cfg_agent in cfg.agents:
        cfg_agent = Config.fromfile(cfg_agent).agent
        agent = AGENTS.build(cfg_agent, default_args={"world": world})
        agents.append(agent)
        world.add_agent(agent)

    # -- command center
    cfg_cc = Config.fromfile(cfg.commandcenter).commandcenter
    commandcenter = AGENTS.build(cfg_cc, default_args={"world": world})

    return world, objects, agents, commandcenter