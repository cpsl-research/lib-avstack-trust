from typing import Any

from avstack.geometry import GlobalOrigin3D


class AgentPipeline:
    """Fusion pipeline for an agent"""

    def __init__(self, sensing, perception, tracking, fusion) -> None:
        self.sensing = {sensor.ID_local: sensor for sensor in sensing}
        self.perception = {percep.ID_local: percep for percep in perception}
        self.tracking = {tracker.ID_local: tracker for tracker in tracking}
        self.fusion = fusion

    def __call__(
        self, platform, tracks_in: dict, world, *args: Any, **kwds: Any
    ) -> list:
        # -- sensing
        s_out = {
            k: v(world.frame, world.t, world.objects) for k, v in self.sensing.items()
        }

        # -- perception
        p_out = {
            k: v(*[s_out[ks] for ks in v.sensor_ID_input])
            for k, v in self.perception.items()
        }

        # -- tracking
        t_out = {
            k: v(
                frame=world.frame,
                t=world.t,
                detections=[p_out[kp] for kp in v.percep_ID_input][
                    0
                ],  # HACK for only 1 percep input...
                platform=platform,
            )
            for k, v in self.tracking.items()
        }

        # -- fusion
        f_out = self.fusion(t_out, tracks_in)

        return f_out


class CommandCenterPipeline:
    """Fusion pipeline for the command center"""

    def __init__(self, clustering, fusion, trust) -> None:
        self.clustering = clustering
        self.fusion = fusion
        self.trust = trust

    def __call__(self, agents: list, tracks_in: dict, *args: Any, **kwds: Any) -> list:
        cluster_inputs = []
        for agent_ID, track in tracks_in.items():
            track.apply(
                "change_reference", reference=GlobalOrigin3D, inplace=True
            )  # eventually inplace=False
            cluster_inputs.append((agent_ID, track))
        clusters = self.clustering(*cluster_inputs)
        fuseds = self.fusion(clusters)
        trusts = self.trust(clusters, fuseds, agents)
        return fuseds, trusts
