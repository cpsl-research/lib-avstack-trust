from typing import Any

from avstack.modules.clustering.clusterers import ClusterSet


class TrustPipeline:
    pass


class PointBasedTrust(TrustPipeline):
    def __init__(self, cluster_scorer, agent_scorer, trust_estimator) -> None:
        self.cluster_scorer = cluster_scorer
        self.agent_scorer = agent_scorer
        self.trust_estimator = trust_estimator

        self.cluster_trusts = {}
        self.agent_trusts = {}

    def __call__(self, clusters: ClusterSet, fuseds: list, agents: list, *args: Any, **kwds: Any) -> Any:
        if len(clusters) > 0:
            import pdb; pdb.set_trace()

        # cluster-based trust measurements
        t_msmts_cluster = [self.cluster_scorer(cluster, agents) for cluster in clusters]

        # cluster-based agent measurements
        t_msmts_agent = [self.agent_scorer(agent, agents, clusters) for agent in agents]

        # trust estimations


        return cluster_trusts, agent_trusts