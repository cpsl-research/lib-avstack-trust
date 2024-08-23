from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.geometry.datastructs import Pose

    from avtrust.fov import Shape


class Agent:
    def __init__(
        self,
        ID: int,
        pose: "Pose",
        fov: "Shape",
        trust_prior: float,
    ) -> None:
        self.ID = ID
        self.pose = pose
        self.fov = fov
        self.trust_prior = trust_prior


class Cluster:
    pass
