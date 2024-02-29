from numpy import pi


_n_objects = 20
_n_agents = 6

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
        "motion": {"type": "Stationary"},
        "communication": {"type": "Omnidirectional"},
        "sensing": [
            {
                "type": "PositionSensor",
                "name": "sensor",
                "fov": {
                    "type": "Wedge",
                    "radius": 20,
                    "angle_start": -pi,
                    "angle_stop": pi,
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
            "clusterer": {"type": "SampledAssignmentClusterer", "assign_radius": 4.0},
            "tracker": {
                "type": "GroupTracker",
                "fusion": {"type": "CovarianceIntersectionFusion"},
                "tracker": {"type": "BasicXyzTracker"},
            },
        },
        "mapping": {"clusterer": ["agents"], "tracker": ["clusterer"]},
    },
}
