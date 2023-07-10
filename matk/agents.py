import itertools
from matk.motion import ConstantSpeedConstantTurn
from avstack.geometry import ReferenceFrame, GlobalOrigin3D


class Object:
    _ids = itertools.count()

    def __init__(self, pose, twist, motion) -> None:
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


class Agent:
    _ids = itertools.count()

    def __init__(self, pose, twist, comms, sensor, tracker, fusion, do_fuse, world) -> None:
        self.ID = next(Agent._ids)
        self.comms = comms
        self.do_fuse = do_fuse
        self.world = world
        self.t = self.world.t
        self.pose = pose
        self.twist = twist
        self.sensor = sensor
        self.tracker = tracker
        self.fusion = fusion
        self._reference = ReferenceFrame(x=self.position.x, v=self.velocity.x,
                                q=self.attitude.q, reference=self.position.reference)

    @property
    def position(self):
        return self.pose.position

    @property
    def attitude(self):
        return self.pose.attitude
    
    @property
    def velocity(self):
        return self.twist.linear
    
    def as_reference(self):
        return self._reference
    
    def update_reference(self, x, v, q):
        self._reference.x = x
        self._reference.v = v
        self._reference.q = q

    def in_range(self, other):
        return self.comms.in_range(self, other)

    def observe(self):
        """Observe environment based on position and world"""
        self.t = self.world.t
        return self.sensor(self.world.frame, self.world.t, self.world.objects)

    def track(self, detections):
        """Run normal tracking on detections"""
        return self.tracker(frame=self.world.frame, t=self.world.t,
                            detections=detections, platform=GlobalOrigin3D)

    def send(self, tracks):
        self.world.receive_tracks(self.t, self.ID, tracks)

    def receive(self):
        """Send information out into the world, receive world information"""
        tracks = self.world.send_tracks(self.t, self.ID)
        return tracks

    def fuse(self, tracks_self, tracks_other):
        """Fuse information from other agents"""
        return self.fusion(tracks_self=tracks_self, tracks_other=tracks_other)

    def plan(self, dt):
        """Plan a path based on the mission"""

    def move(self, dt):
        """Move based on a planned path"""
    

class Radicle(Agent):
    """Untrusted agent

    Mission is to keep monitoring some subregion
    """
    is_root = False

    def __init__(self, pose, twist, comms, sensor, tracker, fusion, do_fuse, world) -> None:
        super().__init__(pose, twist, comms, sensor, tracker, fusion, do_fuse, world)

    def move(self, dt):
        # HACK for now
        self.motion = ConstantSpeedConstantTurn(extent=self.world.extent, radius=5)
        self.pose, self.twist = self.motion.tick(self.pose, self.twist, dt)
        self.update_reference(self.pose.position.x, self.twist.linear.x, self.pose.attitude.q)


class Root(Agent):
    """Trusted agent

    Mission is to monitor the radicle agents and keep awareness
    """

    is_root = True

    def __init__(self, pose, twist, comms, sensor, tracker, fusion, do_fuse, world) -> None:
        super().__init__(pose, twist, comms, sensor, tracker, fusion, do_fuse, world)

    def plan(self, dt):
        pass
