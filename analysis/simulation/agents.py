import itertools
from typing import List

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
        self.t = self.world.t
        self.pose, self.twist, _ = MODELS.build(spawn)(world)

        # set self as a reference
        self._reference = ReferenceFrame(
            x=self.position.x,
            v=self.velocity.x,
            q=self.attitude.q,
            reference=self.position.reference,
        )

        # initialize sensor
        sensors = [
            MODELS.build(
                sensor,
                default_args={"reference": self.as_reference(), "extent": world.extent},
            )
            for sensor in sensing
        ]
        self.sensing = {sensor.ID: sensor for sensor in sensors}

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

    def as_reference(self):
        return self._reference

    def update_reference(self, x, v, q):
        self._reference.x = x
        self._reference.v = v
        self._reference.q = q

    def in_range(self, other):
        return self.comms.in_range(self, other)

    def tick(self):
        self.t = self.world.t
        self.process()

    def process(self):
        # -- sensing
        frame = self.world.frame
        timestamp = self.world.t
        s_out = {
            k: v(frame, timestamp, self.world.objects) for k, v in self.sensing.items()
        }
        tracks_out = self.pipeline(
            sensing=s_out, platform=self._reference, frame=frame, timestamp=timestamp
        )
        self.send(tracks=tracks_out)

    def send(self, tracks):
        """Send tracks to out. Receive from world perspective"""
        self.world.push_tracks(self.t, self.ID, tracks)

    def receive(self):
        """Send information out into the world, receive world information"""
        tracks = self.world.pull_tracks(self.t, self.ID, with_timestamp=False)
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
        self.t = self.world.t
        self.frame = self.world.frame
        self.platform = GlobalOrigin3D
        self.pipeline = PIPELINE.build(pipeline)

    def tick(self):
        self.frame = self.world.frame
        self.t = self.world.t
        agents = self.world.agents
        tracks_in = self.world.pull_tracks(timestamp=self.t, with_timestamp=False)
        tracks_out, cluster_trusts, agent_trusts = self.pipeline(
            agents=agents,
            tracks_in=tracks_in,
            platform=self.platform,
            frame=self.frame,
            timestamp=self.t,
        )
        return tracks_out, cluster_trusts, agent_trusts


@AGENTS.register_module()
class BasicCommandCenter(_CommandCenter):
    pass
