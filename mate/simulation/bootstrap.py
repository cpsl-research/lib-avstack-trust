import numpy as np
from avstack.config import Config
from avstack.modules.clustering import clusterers
from avstack.modules.fusion import track_to_track
from avstack.modules.perception import object3d
from avstack.modules.tracking import tracker2d

from mate import distribution, trust, wrappers
from mate.pipeline import AgentPipeline, CommandCenterPipeline
from mate.simulation import communications, dynamics, sensors
from mate.simulation.agents import Agent, CommandCenter, Object
from mate.simulation.utils import random_pose_twist
from mate.simulation.world import World


def load_scenario_from_config_file(filename: str):
    cfg = Config.fromfile(filename=filename)
    return load_scenario_from_config(cfg)


def load_scenario_from_config(cfg):
    world = load_world_from_config(cfg.world)
    objects = []
    for cfg_obj in cfg.objects:
        obj = load_object_from_config(cfg_obj, world)
        objects.append(obj)
        world.add_object(obj)
    agents = []
    for cfg_agent in cfg.agents:
        agent = load_agent_from_config(cfg_agent, world)
        agents.append(agent)
        world.add_agent(agent)
    commandcenter = load_commandcenter_from_config(cfg.commandcenter, world)

    return world, objects, agents, commandcenter


def load_world_from_config(cfg):
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    return World(dt=cfg.temporal.dt, extent=cfg.spatial.extent)


def load_object_from_config(cfg, world):
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    pose, twist = init_spawn(cfg.initialization.spawn, world)
    motion_model = init_dynamics(cfg.initialization.dynamics, world)

    return Object(
        pose=pose,
        twist=twist,
        motion=motion_model,
    )


def load_agent_from_config(cfg, world):
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    pose, twist = init_spawn(cfg.initialization.spawn, world)
    agent = Agent(pose, twist, cfg.trusted, world)
    dyn = init_dynamics(cfg.initialization.dynamics, world)
    comms, pipeline = init_agent_models(cfg.models, agent, world)

    agent.dynamics = dyn
    agent.comms = comms
    agent.pipeline = pipeline

    return agent


def load_commandcenter_from_config(cfg, world):
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    commandcenter = CommandCenter(world)
    pipeline = init_commandcenter_models(cfg.models, commandcenter, world)
    commandcenter.pipeline = pipeline

    return commandcenter


def init_spawn(spawn, world):
    if spawn.type == "RandomPoseTwist":
        pose, twist, _ = random_pose_twist(extent=world.extent, reference=world.origin)
    else:
        raise NotImplementedError(spawn.type)

    return pose, twist


def init_dynamics(cfg_dynamics, world):
    if cfg_dynamics.type == "ConstantSpeedMarkovTurn":
        dyn = dynamics.ConstantSpeedMarkovTurn(
            extent=world.extent,
            sigma_roll=cfg_dynamics.sigma_roll,
            sigma_pitch=cfg_dynamics.sigma_pitch,
            sigma_yaw=cfg_dynamics.sigma_yaw,
        )
    elif cfg_dynamics.type == "Stationary":
        dyn = dynamics.Stationary(extent=world.extent)
    else:
        raise NotImplementedError(cfg_dynamics.type)

    return dyn


def init_agent_models(cfg, agent, world):
    comms = init_comms(cfg.communication)
    sensing = [init_sensing(cfg_sensing, agent, world) for cfg_sensing in cfg.sensing]
    perception = [init_perception(cfg_percep, world) for cfg_percep in cfg.perception]
    tracking = [init_tracking(cfg_tracker, world) for cfg_tracker in cfg.tracking]
    fusion = init_fusion(cfg.fusion, world)

    pipe = AgentPipeline(sensing, perception, tracking, fusion)

    return comms, pipe


def init_commandcenter_models(cfg, commandcenter, world):
    clustering = init_clustering(cfg.clustering, world)
    fusion = init_fusion(cfg.fusion, world)
    trust = init_trust(cfg.trust, world)
    pipe = CommandCenterPipeline(clustering, fusion, trust)

    return pipe


def init_comms(cfg_comms):
    if cfg_comms.type == "Omnidirectional":
        comms = communications.Omnidirectional(
            max_range=cfg_comms.max_range,
            rate=cfg_comms.rate,
            send=cfg_comms.send,
            receive=cfg_comms.receive,
        )
    else:
        raise NotImplementedError(cfg_comms.type)

    return comms


def init_sensing(cfg_sensing, agent, world):
    if cfg_sensing.type == "PositionSensor":
        sensor = sensors.PositionSensor(
            x=np.array(cfg_sensing.x),
            q=np.quaternion(*cfg_sensing.q),
            reference=agent.as_reference(),
            noise=cfg_sensing.noise,
            extent=world.extent,
            fov=cfg_sensing.fov,
            Pd=cfg_sensing.Pd,
            Dfa=cfg_sensing.Dfa,
        )
    else:
        raise NotImplementedError(cfg_sensing.type)

    sensor = wrappers.SensorWrapper(
        model=sensor,
        ID_local=cfg_sensing.ID_local,
    )

    return sensor


def init_perception(cfg_perception, world):
    if cfg_perception.type == "Passthrough":
        percep = object3d.Passthrough3DObjectDetector()
    else:
        raise NotImplementedError(cfg_perception.type)

    percep = wrappers.PerceptionWrapper(
        model=percep,
        ID_local=cfg_perception.ID_local,
        sensor_ID_input=cfg_perception.sensor_ID_input,
    )

    return percep


def init_tracking(cfg_tracking, world):
    if cfg_tracking.type == "BasicRazTracker":
        tracker = tracker2d.BasicRazTracker(
            threshold_confirmed=cfg_tracking.threshold_confirmed,
            threshold_coast=cfg_tracking.threshold_coast,
        )
    else:
        raise NotImplementedError(cfg_tracking.type)

    tracker = wrappers.TrackingWrapper(
        model=tracker,
        ID_local=cfg_tracking.ID_local,
        percep_ID_input=cfg_tracking.percep_ID_input,
    )

    return tracker


def init_clustering(cfg_cluster, world):
    if cfg_cluster.clustering.type == "SampledAssignmentClustering":
        clust = clusterers.SampledAssignmentClustering(
            assign_radius=cfg_cluster.clustering.assign_radius
        )
    else:
        raise NotImplementedError(cfg_cluster.clustering.type)
    return clust


def init_fusion(cfg_fusion, world):
    if cfg_fusion.type == "AggregatorFusion":
        fuser = track_to_track.AggregatorFusion()
    elif cfg_fusion.type == "NoFusion":
        fuser = track_to_track.NoFusion()
    elif cfg_fusion.type == "CovarianceIntersectionFusion":
        fuser = track_to_track.CovarianceIntersectionFusion()
    else:
        raise NotImplementedError(cfg_fusion.type)
    return fuser


def init_trust(cfg_trust, world):
    if cfg_trust.type == "PointBasedTrust":
        cluster_scorer = trust.measurement.ClusterScorer(
            connective=trust.connectives.StandardFuzzy
        )
        agent_scorer = lambda x, y, z: 0.5
        trust_estimator = trust.estimate.MaximumLikelihoodTrustEstimator(
            dist=distribution.Beta(
                alpha=cfg_trust.estimator.alpha,
                beta=cfg_trust.estimator.beta,
                phi=cfg_trust.estimator.phi,
                lam=cfg_trust.estimator.lam,
            )
        )
        trust_model = trust.PointBasedTrust(
            cluster_scorer, agent_scorer, trust_estimator
        )
    else:
        raise NotImplementedError(cfg_trust.type)
    return trust_model
