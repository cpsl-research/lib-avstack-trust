_n_objects = 20
_n_agents = 10

world = {
    "type": "World",
    "dt": 0.01,
    "dimensions": 3,
    "extent": [[0, 100], [0, 100], [0, 0]],
}

objects = [
    {
        "type": "BasicObject",
        "spawn": {"type": "RandomPoseTwist"},
        "motion": {"type": "ConstantSpeedMarkovTurn"},
    }
    for _ in range(_n_objects)
]

agents = [
    {
        "type": "BasicAgent",
        "trusted": False,
        "spawn": {"type": "RandomPoseTwist"},
        "motion": {"type": "ConstantSpeedConstantTurn"},
        "communication": {"type": "Omnidirectional"},
        "sensing": [{"type": "PositionSensor"}],
        "pipeline": {
            "type": "SerialPipeline",
            "modules": [{"type": "BasicXyzTracker"}],
        },
    }
    for _ in range(_n_agents)
]

commandcenter = {"type": "BasicCommandCenter", "pipeline": {"type": "NoTrustPipeline"}}
