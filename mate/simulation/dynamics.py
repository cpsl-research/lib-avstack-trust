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
        if do_z:
            raise NotImplementedError("Have not implemented z constraint")
        if self.extent is not None:
            adjust = True
            # run into left wall
            if pose.position[0] < self.extent[0][0]:
                pose.position[0] = self.extent[0][0]
                vel_mult = [-1, 1]
            # run into right wall
            elif pose.position[0] > self.extent[0][1]:
                pose.position[0] = self.extent[0][1]
                vel_mult = [-1, 1]
            # run into bottom wall
            elif pose.position[1] < self.extent[1][0]:
                pose.position[1] = self.extent[1][0]
                vel_mult = [1, -1]
            # run into top wall
            elif pose.position[1] > self.extent[1][1]:
                pose.position[1] = self.extent[1][1]
                vel_mult = [1, -1]
            else:
                adjust = False
            # make adjustments
            if adjust:
                twist.linear.x[0] *= vel_mult[0]
                twist.linear.x[1] *= vel_mult[1]
                yaw = np.arctan2(
                    twist.linear[1], twist.linear[0]
                )  # negated bc this is passive transform (??)
                pose.attitude.q = transform_orientation([0, 0, yaw], "euler", "quat")
        return pose, twist


class Stationary(MotionModel):
    def _tick(self, pose, twist, dt):
        return pose, twist


class ConstantSpeedMarkovTurn(MotionModel):
    def __init__(self, extent, sigma_roll=0, sigma_pitch=0, sigma_yaw=1) -> None:
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
        twist.linear.x = new_velocity
        return pose, twist


class ConstantSpeedConstantTurn(MotionModel):
    def __init__(self, extent, radius=10, **kwargs) -> None:
        super().__init__(extent)
        self.radius = radius

    def _tick(self, pose, twist, dt):
        speed = twist.linear.norm()
        dyaw = speed * dt / self.radius
        dq = transform_orientation([0, 0, dyaw], "euler", "quat")
        pose.attitude.q = dq * pose.attitude.q
        new_velocity = speed * pose.attitude.forward_vector
        pose.position = pose.position + dt * (new_velocity + twist.linear.x) / 2
        twist.linear.x = new_velocity
        return pose, twist


class MarkovAcceleration:
    def __init__(self, ax, ay, az) -> None:
        pass
