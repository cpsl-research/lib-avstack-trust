from __future__ import annotations

import numpy as np
from avstack.geometry import GlobalOrigin3D
from avstack.modules.tracking import tracker2d
from utils import _MainThread, do_run, random_pose_twist, run_world_loop

from matk import (
    Object,
    Radicle,
    Root,
    World,
    clustering,
    communications,
    fusion,
    motion,
    sensors,
)


class MainThread(_MainThread):
    def run(self, dt=0.1, print_method="real_time", print_rate=1/2):
        # ===================================================
        # Scenario Parameters
        # ===================================================
        obj_motion_model = motion.ConstantSpeedMarkovTurn(extent=self.extent)

        # -- set up world
        world = World(dt=dt, extent=self.extent)

        # -- spawn objects randomly
        for _ in range(self.n_objects):
            pose, twist, ref = random_pose_twist(extent=self.extent, reference=GlobalOrigin3D)
            obj = Object(
                pose=pose,
                twist=twist,
                motion=obj_motion_model,
            )
            world.add_object(obj)

        # -- spawn radicle agents
        radicles = []
        for _ in range(self.n_radicles):
            pose, twist, ref = random_pose_twist(extent=self.extent, reference=GlobalOrigin3D)
            tracker_radicle = tracker2d.BasicRazTracker(threshold_confirmed=5, threshold_coast=20)
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
                                                    Pd=1.0, Dfa=0.0, extent=self.extent, fov=[30, np.pi/4, np.pi])
            rad.sensor = sensor_radicle
            radicles.append(rad)
            world.add_agent(rad)

        # -- spawn root agent
        pose, twist, ref = random_pose_twist(extent=self.extent, reference=GlobalOrigin3D)
        tracker_root = tracker2d.BasicRazTracker(threshold_confirmed=5, threshold_coast=20)
        clustering_root = clustering.SampledAssignmentClustering(assign_radius=1)
        fusion_root = fusion.CovarianceIntersectionFusion(clustering_root)
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
                                             Pd=1.0, Dfa=0.0, extent=self.extent, fov=[30, np.pi, np.pi])
        root.sensor = sensor_root
        world.add_agent(root)

        run_world_loop(self, world, radicles, root, sleeps=self.sleeps,
            dt=dt, print_method=print_method, print_rate=print_rate)



if __name__ == "__main__":
    do_run(MainThread)