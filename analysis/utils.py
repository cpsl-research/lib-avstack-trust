import argparse
import logging
import random
import sys
import time

import numpy as np
from avstack.geometry import (
    AngularVelocity,
    Attitude,
    Pose,
    Position,
    ReferenceFrame,
    Twist,
    Velocity,
    transform_orientation,
)
from avstack.utils import IterationMonitor
from PyQt5 import QtCore, QtWidgets

from matk import display


def do_run(MainThread):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleeps', default=0.01, type=float)
    parser.add_argument('--display', action="store_true")
    parser.add_argument('--n_radicles', default=6, type=int)
    parser.add_argument('--n_objects', default=10, type=int)
    args = parser.parse_args()

    extent = [[0, 100], [0, 100], [0, 0]]  # x, y, z

    if args.display:
        print('Running with display...')
        app = QtWidgets.QApplication(sys.argv)
        window = display.MainWindow(extent=extent, thread=MainThread(extent, args.sleeps, args.n_radicles, args.n_objects))
        app.exec_()
    else:
        print('Running without display...')
        MainThread(extent, args.sleeps, args.n_radicles, args.n_objects).run()


class _MainThread(QtCore.QThread):
    truth_signal = QtCore.pyqtSignal(int, float, object, object)
    detec_signal = QtCore.pyqtSignal(int, float, object, object)
    estim_signal = QtCore.pyqtSignal(int, float, object, object)

    def __init__(self, extent, sleeps, n_radicles, n_objects, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extent = extent
        self.sleeps = sleeps
        self.n_radicles = n_radicles
        self.n_objects = n_objects

    def run(self):
        raise NotImplementedError


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


def run_world_loop(thread, world, radicles, root, sleeps=0.01, dt=0.1, print_method="real_time", print_rate=1/2):
    # ====================================================================
    # NOTE: this type of world loop only works if the radicle agents
    # are not fusing information between each other. In that context,
    # we will need to solve the track inception problem - meaning,
    # without having sent any tracks, who is the first to establish tracks?
    # ====================================================================
    monitor = IterationMonitor(print_method=print_method, print_rate=print_rate)
    i_iter = 0
    try:
        while True:
            world.tick()

            # Radicles first update their own states and communicate
            # Randomize the order to prevent ordering bias
            random.shuffle(radicles)
            for rad in radicles:
                detections = rad.observe()
                thread.detec_signal.emit(world.frame, world.t, rad, detections)
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
            thread.truth_signal.emit(world.frame, world.t, world.objects, world.agents)
            track_objs = {track.ID:track for track in tracks_out}
            thread.estim_signal.emit(world.frame, world.t, track_objs, world.agents)
            time.sleep(sleeps)
    except (KeyboardInterrupt, Exception) as e:
        logging.warning(e)
    finally:
        pass