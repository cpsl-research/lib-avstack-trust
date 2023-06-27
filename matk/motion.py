import numpy as np
from avstack.geometry import transform_orientation


class MotionModel:
    def __init__(self, extent) -> None:
        self.extent = extent

    def tick(self, pose, twist, dt):
        pose, twist = self._tick(pose, twist, dt)
        pose, twist = self.constrain(pose, twist)
        return pose, twist

    def constrain(self, pose, twist, do_z=False):
        """Constrain the motion of the object to stay within the extent

        The velocity update is slightly buggy. Fix later.
        """
        if self.extent is not None:
            max_dim = 3 if do_z else 2
            for dim in range(max_dim):
                if pose.position[dim] <= self.extent[dim][0]:
                    pose.position[dim] = self.extent[dim][0]
                    dq = transform_orientation([0, 0, np.pi / 2], "euler", "quat")
                    pose.attitude.q = dq * pose.attitude.q
                    speed = twist.linear.norm()
                    twist.linear.vector = speed * pose.attitude.forward_vector
                elif pose.position[dim] > self.extent[dim][1]:
                    pose.position[dim] = self.extent[dim][1]
                    dq = transform_orientation([0, 0, np.pi / 2], "euler", "quat")
                    pose.attitude.q = dq * pose.attitude.q
                    speed = twist.linear.norm()
                    twist.linear.vector = speed * pose.attitude.forward_vector
        return pose, twist


class ConstantSpeedMarkovTurn(MotionModel):
    def __init__(self, extent, sigma_roll=0, sigma_pitch=0, sigma_yaw=1e-2) -> None:
        self.sigma_roll = sigma_roll
        self.sigma_pitch = sigma_pitch
        self.sigma_yaw = sigma_yaw
        self.sigmas = [sigma_roll, sigma_pitch, sigma_yaw]
        super().__init__(extent=extent)

    def _tick(self, pose, twist, dt):
        """Velocity stays the same magnitude, small angle adjustments"""
        speed = twist.linear.norm()
        deul = [dt * sig * np.random.randn() for sig in self.sigmas]
        dq = transform_orientation(deul, "euler", "quat")
        pose.attitude.q = dq * pose.attitude.q
        new_velocity = speed * pose.attitude.forward_vector
        pose.position = pose.position + dt * (new_velocity + twist.linear.x) / 2
        twist.linear.vector = new_velocity
        return pose, twist


class MarkovAcceleration:
    def __init__(self, ax, ay, az) -> None:
        pass
