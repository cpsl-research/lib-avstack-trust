# model for an agent

trusted = False

initialization = dict(
    spawn=dict(type="RandomPoseTwist"),
    dynamics=dict(type="Stationary"),
)

models = dict(
    communication=dict(
        type="Omnidirectional",
        max_range=None,
        rate="continuous",
        send=True,
        receive=False,
    ),
    sensing=[
        dict(
            type="PositionSensor",
            ID_local=0,
            x=[0, 0, 0],
            q=[1, 0, 0, 0],
            noise=[0, 0, 0],
            fov=[30, 0.785398, 3.1415926],  # [30, pi/4, pi]
            Pd=1.0,
            Dfa=0.0,
        )
    ],
    perception=[dict(type="Passthrough", sensor_ID_input=[0], ID_local=0)],
    tracking=[
        dict(
            type="BasicRazTracker",
            ID_local=0,
            percep_ID_input=[0],
            threshold_confirmed=5,
            threshold_coast=20,
        )
    ],
    fusion=dict(type="NoFusion"),
)
