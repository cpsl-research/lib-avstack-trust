import itertools


class Object:
    _ids = itertools.count()

    def __init__(self) -> None:
        self.ID = next(Object._ids)

    def tick(self, dt):
        self.move()

    def move(self):
        pass


class Agent:
    _ids = itertools.count()

    def __init__(self, pose, comms, do_fuse, world) -> None:
        self.ID = next(Agent._ids)
        self.comms = comms
        self.do_fuse = do_fuse
        self.world = world
        self.t = self.world.t
        self.tracks = []
        self.pose = pose

    @property
    def position(self):
        return self.pose.translation

    @property
    def rotation(self):
        return self.pose.rotation

    def in_range(self, other):
        return self.comms.in_range(self, other)

    def observe(self):
        """Observe environment based on position and world"""
        self.t = self.world.t
        detections = []
        return detections

    def track(self, detections):
        """Run normal tracking on detections"""

    def send(self):
        self.world.receive_tracks(self.t, self.ID, self.tracks)

    def receive(self):
        """Send information out into the world, receive world information"""
        tracks = self.world.send_tracks(self.t, self.ID)
        return tracks

    def fuse(self, detections, tracks):
        """Fuse information from other agents"""

    def plan(self):
        """Plan a path based on the mission"""

    def move(self):
        """Move based on a planned path"""


class Radicle(Agent):
    """Untrusted agent

    Mission is to keep monitoring some subregion
    """

    is_root = False

    def __init__(self, pose, comms, do_fuse, world) -> None:
        super().__init__(pose, comms, do_fuse, world)

    def plan(self):
        pass


class Root(Agent):
    """Trusted agent

    Mission is to monitor the radicle agents and keep awareness
    """

    is_root = True

    def __init__(self, pose, comms, do_fuse, world) -> None:
        super().__init__(pose, comms, do_fuse, world)

    def plan(self):
        pass
