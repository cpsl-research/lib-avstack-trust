import random
import sys
import time

import numpy as np
from avstack.geometry import (
    GlobalOrigin3D,
    Pose,
    Position,
    Attitude,
    Twist,
    Velocity,
    AngularVelocity,
    transform_orientation,
)
from avstack.utils import IterationMonitor
from PyQt5 import QtCore, QtWidgets

from matk import Object, Radicle, Root, World, communications, display, motion


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
    return pose, twist


class MainThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(int, float, object, object)

    seed = 1
    np.random.seed(seed)

    def run(self, dt=0.1, n_objects=10, n_radicles=5, print_method="real_time", print_rate=1/2):
        # ===================================================
        # Scenario Parameters
        # ===================================================
        obj_motion_model = motion.ConstantSpeedMarkovTurn(extent=extent)

        # -- set up world
        world = World(dt=dt, extent=extent)

        # -- spawn objects randomly
        for _ in range(n_objects):
            pose, twist = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
            obj = Object(
                pose=pose,
                twist=twist,
                motion=obj_motion_model,
            )
            world.add_object(obj)

        # -- spawn radicle agents
        radicles = []
        for _ in range(n_radicles):
            pose, twist = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
            rad = Radicle(
                pose=pose,
                comms=communications.Omnidirectional(max_range=np.inf),
                do_fuse=False,
                world=world,
            )
            radicles.append(rad)
            world.add_agent(rad)

        # -- spawn root agent  
        pose, twist = random_pose_twist(extent=extent, reference=GlobalOrigin3D)
        root = Root(
            pose=pose,
            comms=communications.Omnidirectional(max_range=np.inf),
            do_fuse=True,
            world=world,
        )
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
                rad.track(detections)
                rad.send()

            # Root agent will receive radicle observations
            detections = root.observe()
            tracks = root.receive()

            # Then root will fuse detections and tracks
            root.fuse(detections, tracks)

            # Now, all agents perform planning and movements
            for rad in radicles:
                rad.plan()
                rad.move()
            root.plan()
            root.move()
            monitor.tick()

            # Update display
            self.signal.emit(world.frame, world.t, world.objects, world.agents)
            time.sleep(sleeps)


if __name__ == "__main__":
    extent = [[0, 100], [0, 100], [0, 0]]  # x, y, z
    sleeps = 0.02

    app = QtWidgets.QApplication(sys.argv)
    window = display.MainWindow(extent=extent, thread=MainThread())
    app.exec_()