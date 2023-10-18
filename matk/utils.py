import random

import numpy as np
from avstack.geometry import (
    AngularVelocity,
    Attitude,
    Pose,
    Position,
    ReferenceFrame,
    Twist,
    Velocity,
    transform_orientation,
)


def random_pose_twist(extent, reference, vmin=2, vmax=5, vsig=2, buffer=10):
    x = np.array(
        [
            random.uniform(ext[0] + buffer, ext[1] - buffer)
            if ext[0] < ext[1]
            else ext[0]
            for ext in extent
        ]
    )
    q = transform_orientation([0, 0, np.random.uniform(0, 2 * np.pi)], "euler", "quat")
    loc = Position(x, reference)
    rot = Attitude(q, reference)
    pose = Pose(loc, rot)
    v = vsig * np.random.randn()
    if abs(v) > vmax:
        v = np.sign(v) * vmax
    elif abs(v) < vmin:
        v = np.sign(v) * vmin
    linear = Velocity(v * rot.forward_vector, reference)
    angular = AngularVelocity(np.quaternion(1), reference)
    twist = Twist(linear, angular)
    point_as_ref = ReferenceFrame(
        x=pose.position.x, v=twist.linear.x, q=pose.attitude.q, reference=reference
    )
    return pose, twist, point_as_ref
