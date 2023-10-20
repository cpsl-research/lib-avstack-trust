# model for an aggregate scenario

world = "../config/_base_/base_world.py"

n_agents = 5
agents = ["../config/_base_/base_agent.py" for _ in range(n_agents)]

n_objects = 30
objects = ["../config/_base_/base_object.py" for _ in range(n_objects)]

commandcenter = "../config/_base_/base_command_center.py"
