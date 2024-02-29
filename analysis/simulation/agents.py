import itertools
from typing import List

import numpy as np
from avstack.config import AGENTS, MODELS, PIPELINE, ConfigDict
from avstack.geometry import GlobalOrigin3D, ReferenceFrame


class _Object:
    _ids = itertools.count()

    def __init__(self, spawn, motion, world, ID_force=None) -> None:
        if ID_force:
            self.ID = ID_force
        else:
            self.ID = next(_Object._ids)
        if world is None:
            raise TypeError("Need to set world before initializing")
        self.pose, self.twist, _ = MODELS.build(spawn)(world)
        motion.extent = world.extent
        self.motion = MODELS.build(motion)

    @property
    def position(self):
        return self.pose.position

    @property
    def rotation(self):
        return self.pose.rotation

    @property
    def velocity(self):
        return self.twist.linear

    def tick(self, dt):
        self.pose, self.twist = self.motion.tick(self.pose, self.twist, dt)

    def change_reference(self, reference, inplace):
        if inplace:
            self.pose.change_reference(reference, inplace=inplace)
            self.twist.change_reference(reference, inplace=inplace)
        else:
            pose = self.pose.change_reference(reference, inplace=inplace)
            twist = self.twist.change_reference(reference, inplace=inplace)
            return _Object(pose, twist, self.motion, ID_force=self.ID)


@AGENTS.register_module()
class BasicObject(_Object):
    pass


class _Agent:
    _ids = itertools.count()

    def __init__(
        self,
        trusted,
        spawn,
        motion,
        communication,
        sensing: List[ConfigDict],
        pipeline,
        world,
    ) -> None:
        self.ID = next(_Agent._ids)
        self.trusted = trusted
        self._trust = 1.0 if trusted else 0.5
        self.world = world
        self.timestamp = self.world.timestamp
        # initialize reference
        self._reference = ReferenceFrame(
            x=np.zeros((3,)),
            v=np.zeros((3,)),
            q=np.quaternion(1),
            reference=GlobalOrigin3D,
        )
        self.pose, self.twist, _ = MODELS.build(spawn)(world)

        # initialize sensor
        sensors = [
            MODELS.build(
                sensor,
                default_args={"reference": self.as_reference(), "extent": world.extent},
            )
            for sensor in sensing
        ]
        self.sensing = {sensor.name: sensor for sensor in sensors}
        assert len(self.sensing) == len(sensors)

        # initialize pipeline
        motion.extent = world.extent
        self.motion = MODELS.build(motion)
        self.comms = MODELS.build(communication)
        self.pipeline = PIPELINE.build(pipeline)

    @property
    def position(self):
        return self.pose.position

    @property
    def attitude(self):
        return self.pose.attitude

    @property
    def velocity(self):
        return self.twist.linear

    @property
    def trust(self):
        return self._trust

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose
        self._reference.x = pose.position.x
        self._reference.q = pose.attitude.q

    @property
    def twist(self):
        return self._twist

    @twist.setter
    def twist(self, twist):
        self._twist = twist
        self._reference.v = twist.linear.x

    def as_reference(self):
        return self._reference

    def in_range(self, other):
        return self.comms.in_range(self, other)

    def tick(self):
        dt = self.world.timestamp - self.timestamp
        self.timestamp = self.world.timestamp
        self.process()
        self.pose, self.twist = self.motion.tick(self.pose, self.twist, dt)

    def process(self):
        # -- sensing
        frame = self.world.frame
        timestamp = self.world.timestamp
        s_out = {
            k: v(frame, timestamp, self.world.objects) for k, v in self.sensing.items()
        }
        tracks_out = self.pipeline(
            data=s_out,
            platform=self._reference,
        )
        self.send(tracks=tracks_out)

    def send(self, tracks):
        """Send tracks to out. Receive from world perspective"""
        self.world.push_agent_data(self.timestamp, self.ID, tracks)

    def receive(self):
        """Send information out into the world, receive world information"""
        tracks = self.world.pull_agent_data(
            self.timestamp, self.ID, with_timestamp=False
        )
        return tracks

    def plan(self):
        """Plan a path based on the mission"""

    def move(self):
        """Move based on a planned path"""


@AGENTS.register_module()
class BasicAgent(_Agent):
    pass


class _CommandCenter:
    ID = -1  # special ID for the command center

    def __init__(self, pipeline, world) -> None:
        self.world = world
        self.timestamp = self.world.timestamp
        self.frame = self.world.frame
        self.platform = GlobalOrigin3D
        self.pipeline = PIPELINE.build(pipeline)

    def tick(self):
        self.frame = self.world.frame
        self.timestamp = self.world.timestamp
        fovs = {
            agent.ID: list(agent.sensing.values())[0].fov for agent in self.world.agents
        }
        platforms = {
            agent.ID: list(agent.sensing.values())[0]._reference
            for agent in self.world.agents
        }
        agent_data = self.world.pull_agent_data(
            timestamp=self.timestamp,
            with_timestamp=False,
            target_reference=GlobalOrigin3D,
        )
        output = self.pipeline(
            data={"agents": agent_data, "fovs": fovs, "platforms": platforms},
        )
        return output


@AGENTS.register_module()
class BasicCommandCenter(_CommandCenter):
    pass
