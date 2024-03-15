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
                "Pd": 0.95,
                "Dfa": 1e-4,
                "Pp_FP": 0,
                "Pp_FN": 0,
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
        "type": "TrustBasedFusionPipeline",
        "trust_pipeline": {
            "type": "MappedPipeline",
            "modules": {
                "psms": None,
                "estimator": None,
            },
            "mapping": {"psms": ["tracks", "input_data", "fovs", "platforms"], "estimator": ["psms"]}
        },
        "fusion_pipeline": {
            "type": "MappedPipeline",
            "modules": {
                "tracker": {
                    "type": "MeasurementBasedMultiTracker",
                    "tracker": {"type": "BasicXyzTracker"},
                },
            },
            "mapping": {"tracker": ["input_data", "fovs", "platforms"]},
        },
    },
}


# commandcenter = {
#     "type": "BasicCommandCenter",
#     "pipeline": {
#         "type": "MappedPipeline",
#         "modules": {
#             "tracker": {
#                 "type": "MeasurementBasedMultiTracker",
#                 "tracker": {"type": "BasicXyzTracker"},
#             },
#         },
#         "mapping": {"tracker": ["input_data", "fovs", "platforms"]},
#     },
# }
