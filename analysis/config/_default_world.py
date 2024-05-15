world = {
    "type": "World",
    "dt": 0.01,
    "dimensions": 3,
    "extent": [[0, 100], [0, 100], [0, 0]],
}

_n_objects = 20
objects = [
    {
        "type": "BasicObject",
        "spawn": {"type": "RandomPoseTwist"},
        "motion": {"type": "ConstantSpeedMarkovTurn"},
    }
    for _ in range(_n_objects)
]
