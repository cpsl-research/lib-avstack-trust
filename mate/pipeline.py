from typing import Any, List

from avstack.config import MODELS, PIPELINE
from avstack.modules.tracking.tracks import GroupTrack


class _TrustPipeline:
    def __init__(self, *args, **kwds):
        pass


@PIPELINE.register_module()
class NoTrustPipeline(_TrustPipeline):
    def __call__(
        self, group_tracks: List[GroupTrack], agents: list, *args: Any, **kwds: Any
    ) -> Any:
        return [None] * len(group_tracks), [None] * len(agents)


@PIPELINE.register_module()
class PointBasedTrustPipeline(_TrustPipeline):
    def __init__(
        self, cluster_scorer, agent_scorer, trust_estimator, *args, **kwds
    ) -> None:
        # algorithms
        self.cluster_scorer = MODELS.build(cluster_scorer)
        self.agent_scorer = MODELS.build(agent_scorer)
        self.cluster_trust_estimator = trust_estimator
        self.agent_trust_estimator = trust_estimator

        # data structures
        self.cluster_trusts = {}
        self.agent_trusts = {}

    def __call__(
        self,
        group_tracks: List[GroupTrack],
        agents: list,
        timestamp: float,
        *args: Any,
        **kwds: Any
    ) -> Any:

        # clear any out of date ones
        IDs_remove = []
        for ID in self.cluster_trusts.keys():
            for track in group_tracks:
                if ID == track.ID:
                    break
            else:
                # remove this distribution if we not loner have the group track
                IDs_remove.append(ID)
        for ID in IDs_remove:
            self.cluster_trusts.pop(ID)

        # cluster-based trust measurements
        for group_track in group_tracks:
            trust_msmt_cluster = self.cluster_scorer(group_track, agents)
            if group_track.ID not in self.cluster_trusts:
                self.cluster_trusts[group_track.ID] = MODELS.build(
                    self.cluster_trust_estimator
                )
            self.cluster_trusts[group_track.ID].update(
                timestamp=group_track.t,
                trust=trust_msmt_cluster,
            )

        # propagate all cluster-based trusts to the present time
        for cluster_trust in self.cluster_trusts.values():
            cluster_trust.propagate(timestamp)

        # cluster-based agent measurements
        for agent in agents:
            trust_msmt_agent = self.agent_scorer(agent, agents, group_tracks)
            if agent.ID not in self.agent_trusts:
                self.agent_trusts[agent.ID] = MODELS.build(self.agent_trust_estimator)
            self.agent_trusts[agent.ID].update(
                timestamp=timestamp, trust=trust_msmt_agent
            )

        return self.cluster_trusts, self.agent_trusts
