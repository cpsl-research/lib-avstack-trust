from avstack.geometry.datastructs import Pose

from mate.fov import FieldOfView


class Agent:
    def __init__(
        self,
        ID: int,
        pose: Pose,
        fov: FieldOfView,
        trust_prior: float,
    ) -> None:
        self.ID = ID
        self.pose = pose
        self.fov = fov
        self.trust_prior = trust_prior


class Cluster:
    pass
