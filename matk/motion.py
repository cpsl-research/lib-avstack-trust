import numpy as np
from avstack.geometry import transform_orientation


class MotionModel:
    def __init__(self) -> None:
        pass


class ConstantSpeedMarkovTurn:
    def __init__(self, sigma_roll=1e-2, sigma_pitch=0, sigma_yaw=1e-2) -> None:
        self.sigma_roll = sigma_roll
        self.sigma_pitch = sigma_pitch
        self.sigma_yaw = sigma_yaw
        self.sigmas = [sigma_roll, sigma_pitch, sigma_yaw]

    def tick(self, pose, twist, dt):
        """Velocity stays the same magnitude, small angle adjustments"""
        speed = twist.linear.norm()
        deul = [dt*sig*np.random.randn() for sig in self.sigmas]
        dq = transform_orientation(deul, 'euler', 'quat')
        pose.rotation.q = dq * pose.rotation.q
        new_velocity = speed * pose.rotation.forward_vector
        pose.position = pose.position + dt*(new_velocity + twist.linear)/2
        twist.linear.vector = new_velocity
        return pose, twist


class MarkovAcceleration:
    def __init__(self, ax, ay, az) -> None:
        pass