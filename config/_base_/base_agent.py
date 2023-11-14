# model for an agent

agent = dict(
    type="Agent",
    trusted=False,
    spawn=dict(type="RandomPoseTwist"),
    motion=dict(type="Stationary"),
    communication=dict(
        type="Omnidirectional",
        max_range=None,
        rate="continuous",
        send=True,
        receive=False,
    ),
    sensing=[
        dict(
            type="SensorWrapper",
            ID=0,
            model=dict(
                type="PositionSensor",
                ID=0,
                x=[0, 0, 0],
                q=[1, 0, 0, 0],
                noise=[0, 0, 0],
                fov=[30, 0.785398, 3.1415926],  # [30, pi/4, pi]
                Pd=1.0,
                Dfa=0.0,
            ),
        )
    ],
    pipeline=dict(
        type="AgentPipeline",
        perception=[
            dict(
                type="PerceptionWrapper",
                ID=0,
                ID_input=[0],
                algorithm=dict(type="Passthrough3DObjectDetector"),
            )
        ],
        tracking=[
            dict(
                type="TrackingWrapper",
                ID=0,
                ID_input=[0],
                algorithm=dict(
                    type="BasicRazTracker",
                    threshold_confirmed=5,
                    threshold_coast=20,
                ),
            )
        ],
    ),
)
