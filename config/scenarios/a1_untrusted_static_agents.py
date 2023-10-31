# model for an aggregate scenario

world = "../config/_base_/base_world.py"

n_untrusted_agents = 5
agents = [
    "../config/agent/untrusted_static_agent.py" for _ in range(n_untrusted_agents)
]

n_objects = 30
objects = ["../config/_base_/base_object.py" for _ in range(n_objects)]

commandcenter = "../config/command_center/point_based_ci_fusion.py"
