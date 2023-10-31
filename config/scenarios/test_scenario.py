# model for an aggregate scenario

world = "../config/_base_/base_world.py"

n_untrusted_agents = 5
n_trusted_agents = 3
_untrusted_agents = [
    "../config/_base_/base_untrusted_agent.py" for _ in range(n_untrusted_agents)
]
_trusted_agents = [
    "../config/_base_/base_trusted_agent.py" for _ in range(n_trusted_agents)
]
agents = _untrusted_agents + _trusted_agents

n_objects = 30
objects = ["../config/_base_/base_object.py" for _ in range(n_objects)]

commandcenter = "../config/_base_/base_command_center.py"
