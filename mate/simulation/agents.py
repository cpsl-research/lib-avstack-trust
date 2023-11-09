import itertools

from avstack.geometry import ReferenceFrame


class Object:
    _ids = itertools.count()

    def __init__(self, pose, twist, motion, ID_force=None) -> None:
        if ID_force:
            self.ID = ID_force
        else:
            self.ID = next(Object._ids)
        self.pose = pose
        self.twist = twist
        self.motion = motion

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
            return Object(pose, twist, self.motion, ID_force=self.ID)


class Agent:
    _ids = itertools.count()

    def __init__(self, pose, twist, trusted, world) -> None:
        self.ID = next(Agent._ids)
        self.trusted = trusted
        self._trust = 1.0 if trusted else 0.5
        self.world = world
        self.t = self.world.t
        self.pose = pose
        self.twist = twist
        self._reference = ReferenceFrame(
            x=self.position.x,
            v=self.velocity.x,
            q=self.attitude.q,
            reference=self.position.reference,
        )
        self.comms = None
        self.pipeline = None

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
        tracks_in = self.receive()
        tracks_out = self.pipeline(
            platform=self._reference, tracks_in=tracks_in, world=self.world
        )
        self.send(tracks=tracks_out)

    def send(self, tracks):
        """Send tracks to out. Receive from world perspective"""
        self.world.push_tracks(self.t, self.ID, tracks)

    def receive(self):
        """Send information out into the world, receive world information"""
        tracks = self.world.pull_tracks(self.t, self.ID)
        return tracks

    def plan(self):
        """Plan a path based on the mission"""

    def move(self):
        """Move based on a planned path"""


class CommandCenter:
    ID = -1  # special ID for the command center

    def __init__(self, world) -> None:
        self.world = world
        self.t = self.world.t

    def tick(self):
        self.t = self.world.t
        agents = self.world.agents
        tracks_in = self.world.pull_tracks(self.t, self.ID, with_timestamp=False)
        tracks_out = self.pipeline(agents=agents, tracks_in=tracks_in)
        return tracks_out
