from numpy import pi


_base_ = ["_default_world.py"]

_n_agents = 2

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
            "modules": {"extractor": lambda x, *args, **kwargs: x},
            "mapping": {"extractor": ["sensor"]},
        },
    }
    for _ in range(_n_agents)
]

commandcenter = {
    "type": "BasicCommandCenter",
    "pipeline": {
        "type": "MappedPipeline",
        "modules": {
            "tracker": {
                "type": "MeasurementBasedMultiTracker",
                "tracker": {"type": "BasicXyzTracker"},
            },
        },
        "mapping": {"tracker": ["agents", "fovs", "platforms"]},
    },
}