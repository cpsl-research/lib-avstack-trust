from typing import TYPE_CHECKING, Dict, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.environment.objects import ObjectState
    from avstack.geometry import Polygon
    from avtrust.estimator import TrustEstimator

from avstack.config import HOOKS

from avtrust.config import AVTRUST


@HOOKS.register_module()
class TrustFusionHook:
    def __init__(
        self,
        model: Union[dict, "TrustEstimator"],
        n_calls_burnin: int = 0,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.i_calls = 0
        self.n_calls_burnin = n_calls_burnin
        self.model = AVTRUST.build(model) if isinstance(model, dict) else model
        self.verbose = verbose
        self.trust_agents = None
        self.trust_tracks = None
        self.psms_agents = None
        self.psms_tracks = None

    def __call__(
        self,
        agents: Dict[str, "ObjectState"],
        fov_agents: Dict[str, "Polygon"],
        tracks_agents: Dict[str, "DataContainer"],
        tracks_fused: "DataContainer",
        logger=None,
        *args,
        **kwargs,
    ):
        self.ran_trust = False
        if len(fov_agents) > 0:
            # only run if we've exceeded the burnin
            self.i_calls += 1
            if self.i_calls > self.n_calls_burnin:
                position_agents = {k: agent.position for k, agent in agents.items()}
                (
                    self.trust_agents,
                    self.trust_tracks,
                    self.psms_agents,
                    self.psms_tracks,
                ) = self.model(
                    position_agents=position_agents,
                    fov_agents=fov_agents,
                    tracks_agents=tracks_agents,
                    tracks_cc=tracks_fused,
                )
                self.ran_trust = True
                if self.verbose:
                    if logger is not None:
                        logger.info(f"Ran trust model at time {tracks_fused.timestamp}")
                        logger.info(
                            f"Maintaining trust for agents {list(self.trust_agents.keys())}"
                            f" and tracks {list(self.trust_tracks.keys())}"
                        )
        else:
            self.psms_agents = None
            self.psms_tracks = None
