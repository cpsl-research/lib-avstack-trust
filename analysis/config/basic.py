from numpy import pi


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
        "sensing": [
            {
                "type": "PositionSensor",
                "name": "sensor",
                "fov": {
                    "type": "Wedge",
                    "radius": 20,
                    "angle_start": -pi / 4,
                    "angle_stop": pi / 4,
                },
            }
        ],
        "pipeline": {
            "type": "MappedPipeline",
            "modules": {"tracker": {"type": "BasicXyzTracker"}},
            "mapping": {"tracker": ["sensor"]},
        },
    }
    for _ in range(_n_agents)
]

commandcenter = {
    "type": "BasicCommandCenter",
    "pipeline": {
        "type": "MappedPipeline",
        "modules": {
            "clusterer": {"type": "SampledAssignmentClusterer"},
            "tracker": {
                "type": "GroupTracker",
                "fusion": {"type": "CovarianceIntersectionFusion"},
                "tracker": {"type": "BasicXyzTracker"},
            },
        },
        "mapping": {"clusterer": ["agents"], "tracker": ["clusterer"]},
    },
}
