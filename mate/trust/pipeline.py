from typing import Any

from avstack.config import ALGORITHMS, PIPELINE
from avstack.modules.clustering.clusterers import ClusterSet


class _TrustPipeline:
    pass


@PIPELINE.register_module()
class NoTrustPipeline(_TrustPipeline):
    def __call__(
        self, clusters: ClusterSet, agents: list, *args: Any, **kwds: Any
    ) -> Any:
        return [None] * len(clusters), [None] * len(agents)


@PIPELINE.register_module()
class PointBasedTrustPipeline(_TrustPipeline):
    def __init__(
        self, cluster_scorer, agent_scorer, trust_estimator, *args, **kwds
    ) -> None:
        # algorithms
        self.cluster_scorer = ALGORITHMS.build(cluster_scorer)
        self.agent_scorer = ALGORITHMS.build(agent_scorer)
        self.trust_estimator = ALGORITHMS.build(trust_estimator)

        # data structures
        self.cluster_trusts = {}
        self.agent_trusts = {}

    def __call__(
        self, clusters: ClusterSet, agents: list, *args: Any, **kwds: Any
    ) -> Any:

        # cluster-based trust measurements
        t_msmts_cluster = [self.cluster_scorer(cluster, agents) for cluster in clusters]

        # cluster-based agent measurements
        t_msmts_agent = [self.agent_scorer(agent, agents, clusters) for agent in agents]

        # trust estimations
        cluster_trusts = [None] * len(clusters)
        agent_trusts = [None] * len(agents)

        return cluster_trusts, agent_trusts
