import sys
import random
import numpy as np
import time

from PyQt5 import QtCore, QtWidgets
from matk import Object, Radicle, Root, World
from matk import communications, motion, display
from avstack.utils import IterationMonitor
from avstack.geometry import NominalOriginStandard, Pose, Twist, VectorDirMag, Translation, Rotation, transform_orientation


def random_pose_twist(extent, origin):
    x = np.array([random.uniform(ext[0], ext[1]) if ext[0]<ext[1] else ext[0] for ext in extent])
    q = transform_orientation(np.random.uniform(0, 2*np.pi, 3), 'euler', 'quat')
    loc = Translation(x, origin)
    rot = Rotation(q, origin)
    pose = Pose(loc, rot)
    v = max(-5, min(5, 2*np.random.randn()))
    linear = VectorDirMag(v*rot.forward_vector, origin)
    angular = VectorDirMag(np.zeros((3,)), origin)
    twist = Twist(linear, angular)
    return pose, twist


class MainThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(object, object)

    def run(self):
        # ===================================================
        # Scenario Parameters
        # ===================================================
        vmax = 5  # m/s
        dt = 0.10  # seconds
        obj_motion_model = motion.ConstantSpeedMarkovTurn(extent=extent)

        # -- set up world
        world = World(dt=dt, extent=extent)

        # -- spawn objects randomly
        n_objects = 10
        for _ in range(n_objects):
            pose, twist = random_pose_twist(extent=extent, origin=NominalOriginStandard)
            obj = Object(
                pose=pose,
                twist=twist,
                motion=obj_motion_model,
            )
            world.add_object(obj)

        # -- spawn radicle agents
        n_radicles = 5
        radicles = []
        for _ in range(n_radicles):
            pose, twist = random_pose_twist(extent=extent, origin=NominalOriginStandard)
            rad = Radicle(
                pose=pose,
                comms=communications.Omnidirectional(max_range=np.inf),
                do_fuse=False,
                world=world,
            )
            radicles.append(rad)
            world.add_agent(rad)

        # -- spawn root agent  
        pose, twist = random_pose_twist(extent=extent, origin=NominalOriginStandard)
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
        monitor = IterationMonitor(sim_dt=dt, print_method="real_time", print_rate=1/2)
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
            self.signal.emit(world.objects, world.agents)
            time.sleep(sleeps)


if __name__ == "__main__":
    extent = [[0, 100], [0, 100], [0, 0]]  # x, y, z
    sleeps = 0.02

    app = QtWidgets.QApplication(sys.argv)
    window = display.MainWindow(extent=extent, thread=MainThread())
    app.exec_()