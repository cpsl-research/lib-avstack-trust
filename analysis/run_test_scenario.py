import random
import numpy as np
from matk import Object, Radicle, Root, World
from matk import communications
from avstack.utils import IterationMonitor
from avstack.geometry import NominalTransform


if __name__ == "__main__":
    dt = 0.10
    world = World(dt=dt)
    default_pose = NominalTransform

    # -- spawn objects
    n_objects = 10
    for _ in range(n_objects):
        obj = Object()

    # -- spawn radicle agents
    n_radicles = 5
    radicles = []
    for _ in range(n_radicles):
        rad = Radicle(
            pose=default_pose,
            comms=communications.Omnidirectional(max_range=np.inf),
            do_fuse=False,
            world=world,
        )
        radicles.append(rad)
        world.add_agent(rad)

    # -- spawn root agent    
    root = Root(
        pose=default_pose,
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
