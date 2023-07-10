import random
import sys
import time

import numpy as np
from avstack.geometry import (
    GlobalOrigin3D,
    ReferenceFrame,
    Pose,
    Position,
    Attitude,
    Twist,
    Velocity,
    AngularVelocity,
    transform_orientation,
)
from avstack.modules.tracking.tracker2d import BasicRazTracker
from avstack.utils import IterationMonitor
from PyQt5 import QtCore, QtWidgets

from matk import Object, Radicle, Root, World, communications, display, motion, sensors, fusion


def random_pose_twist(extent, reference, vmin=2, vmax=5, vsig=2, buffer=10):
    x = np.array([random.uniform(ext[0]+buffer, ext[1]-buffer) if ext[0]<ext[1] else ext[0] for ext in extent])
    q = transform_orientation([0, 0, np.random.uniform(0, 2*np.pi)], 'euler', 'quat')
    loc = Position(x, reference)
    rot = Attitude(q, reference)
    pose = Pose(loc, rot)
    v = vsig*np.random.randn()
    if abs(v) > vmax:
        v = np.sign(v) * vmax
    elif abs(v) < vmin:
        v = np.sign(v) * vmin
    linear = Velocity(v*rot.forward_vector, reference)
    angular = AngularVelocity(np.quaternion(1), reference)
    twist = Twist(linear, angular)
    ref = ReferenceFrame(x=pose.position.x, v=twist.linear.x, q=pose.attitude.q, reference=reference)
    return pose, twist, ref


class MainThread(QtCore.QThread):
    truth_signal = QtCore.pyqtSignal(int, float, object, object)
    estim_signal = QtCore.pyqtSignal(int, float, object, object)

    seed = 1
    np.random.seed(seed)

    def run(self, dt=0.1, n_objects=10, n_radicles=6, print_method="real_time", print_rate=1/2):
        # ===================================================
        # Scenario Parameters
        # ===================================================
        obj_motion_model = motion.ConstantSpeedMarkovTurn(extent=extent)

        # -- set up world
        world = World(dt=dt, extent=extent)

        # -- spawn objects randomly
        for _ in range(n_objects):
            pose, twist, ref = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
            obj = Object(
                pose=pose,
                twist=twist,
                motion=obj_motion_model,
            )
            world.add_object(obj)

        # -- spawn radicle agents
        radicles = []
        for _ in range(n_radicles):
            pose, twist, ref = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
            tracker_radicle = BasicRazTracker()
            fusion_radicle = None
            rad = Radicle(
                pose=pose,
                twist=twist,
                comms=communications.Omnidirectional(max_range=np.inf),
                sensor=None,
                tracker=tracker_radicle,
                fusion=fusion_radicle,
                do_fuse=False,
                world=world,
            )
            x = np.array([0,0,0])
            q = np.quaternion(1)
            sensor_radicle = sensors.PositionSensor(x, q, rad.as_reference(), noise=[0, 0, 0],
                                                    extent=extent, fov=[30, np.pi/180*30, np.pi])
            rad.sensor = sensor_radicle
            radicles.append(rad)
            world.add_agent(rad)

        # -- spawn root agent
        pose, twist, ref = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
        tracker_root = BasicRazTracker()
        fusion_root = fusion.AggregatorFusion()
        root = Root(
            pose=pose,
            twist=twist,
            comms=communications.Omnidirectional(max_range=np.inf),
            sensor=None,
            tracker=tracker_root,
            fusion=fusion_root,
            do_fuse=True,
            world=world,
        )
        x = np.array([0,0,0])
        q = np.quaternion(1)
        sensor_root = sensors.PositionSensor(x, q, root.as_reference(), noise=[0, 0, 0],
                                             extent=extent, fov=[30, np.pi/180*30, np.pi])
        root.sensor = sensor_root
        world.add_agent(root)

        # -- run world loop
        # ====================================================================
        # NOTE: this type of world loop only works if the radicle agents
        # are not fusing information between each other. In that context,
        # we will need to solve the track inception problem - meaning,
        # without having sent any tracks, who is the first to establish tracks?
        # ====================================================================
        monitor = IterationMonitor(sim_dt=dt, print_method=print_method, print_rate=print_rate)
        i_iter = 0
        while True:
            world.tick()

            # Radicles first update their own states and communicate
            # Randomize the order to prevent ordering bias
            random.shuffle(radicles)
            for rad in radicles:
                detections = rad.observe()
                tracks = rad.track(detections)
                rad.send(tracks)

            # Root agent will receive radicle observations
            detections = root.observe()
            tracks_root = root.track(detections)

            # Then root will fuse detections and tracks
            tracks_other = root.receive()
            tracks_out = root.fuse(tracks_root, tracks_other)

            # Now, all agents perform planning and movements
            for rad in radicles:
                rad.plan(dt)
                rad.move(dt)
            root.plan(dt)
            root.move(dt)
            monitor.tick()

            # Update displays
            self.truth_signal.emit(world.frame, world.t, world.objects, world.agents)
            track_objs = {track.ID:track for track in tracks_out}
            self.estim_signal.emit(world.frame, world.t, track_objs, world.agents)
            time.sleep(sleeps)


if __name__ == "__main__":
    extent = [[0, 100], [0, 100], [0, 0]]  # x, y, z
    sleeps = 0.02

    app = QtWidgets.QApplication(sys.argv)
    window = display.MainWindow(extent=extent, thread=MainThread())
    app.exec_()