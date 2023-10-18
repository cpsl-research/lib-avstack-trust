# model for an object in the scene

initialization = dict(
    spawn=dict(type="RandomPoseTwist"),
    dynamics=dict(
        type="ConstantSpeedMarkovTurn",
        sigma_roll=0,
        sigma_pitch=0,
        sigma_yaw=1,
    ),
)
