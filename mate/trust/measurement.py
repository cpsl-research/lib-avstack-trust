import math
from typing import Any

import numpy as np
from avstack.config import ALGORITHMS, MODELS

from mate import connectives, distribution


@ALGORITHMS.register_module()
class AgentScorer:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return 0.5


@ALGORITHMS.register_module()
class ClusterScorer:
    def __init__(self, connective: connectives.StandardFuzzy) -> None:
        self.connective = MODELS.build(connective)

    def norm(self, a, b):
        return self.connective.norm(a, b)

    def conorm(self, a, b):
        return self.connective.conorm(a, b)

    def negation(self, a):
        return self.connective.negation(a)

    def __call__(self, group_track, agents, *args: Any, **kwds: Any) -> Any:
        """Obtain a trust score for the cluster of interest

        trust represents the probability that the object is real and that the state is trusted
        """

        fused_state = group_track.state
        cluster = group_track.members

        trusted = [agent_is_trusted(agent) for agent in agents]
        expected = [agent_expected_to_see(agent, cluster) for agent in agents]
        scores = [agent_high_track_score(agent, cluster) for agent in agents]
        beliefs = [self.norm(expected[i], scores[i]) for i in range(len(agents))]
        trusted_and_believes = [
            self.norm(trusted[i], beliefs[i]) for i in range(len(agents))
        ]
        similars = [
            agent_has_similar_track_state(agent, fused_state, cluster)
            for agent in agents
        ]
        similar_and_believes = [
            self.norm(similars[i], beliefs[i]) for i in range(len(agents))
        ]

        # =======================
        # cluster existence
        # =======================

        # ---- no trusted agent doesn't believe the cluster exists
        # ---- at least 1 trusted agent believes exists
        no_trusted_dont_believe = 1.0
        a_trusted_believes = 0.0
        for i in range(len(agents)):
            # -- no trusteds don't belief
            no_trusted_dont_believe = self.norm(
                no_trusted_dont_believe, trusted_and_believes[i]
            )
            # -- yes trusted believes belief
            a_trusted_believes = self.conorm(
                a_trusted_believes, trusted_and_believes[i]
            )

        # ---- many agents believe exists
        belief_sorted = sorted(beliefs, reverse=True)
        many_agents_believe = 0.0
        for i_agents in range(2, len(agents)):  # 1 will never be "many"
            # -- aggregate the beliefs (e.g., get the "worst best" in the standard case)
            top_i_beliefs = belief_sorted[:i_agents]
            agg_belief = 1.0
            for belief in top_i_beliefs:
                agg_belief = self.norm(agg_belief, belief)
            # -- determine if i is many
            i_is_many = self.conorm(
                i_agents_is_many(i_agents),
                i_agents_is_large_fraction(
                    i_agents, sum([ex > 0.5 for ex in expected])
                ),
            )
            # -- combine
            many_agents_believe = self.conorm(
                many_agents_believe, self.norm(agg_belief, i_is_many)
            )

        # ---- aggregate cluster existence scores
        cluster_exists = self.norm(
            no_trusted_dont_believe,
            self.conorm(a_trusted_believes, many_agents_believe),
        )

        # =======================
        # cluster state agreement
        # =======================

        # ---- a trusted agent has a similar state
        a_trusted_has_similar = 0.0
        for i in range(len(agents)):
            a_trusted_has_similar = self.conorm(
                a_trusted_has_similar, self.norm(trusted_and_believes[i], similars[i])
            )

        # ---- many agents have similar states
        similar_and_believes_sorted = sorted(similar_and_believes, reverse=True)
        many_agents_believe_and_similar = 0.0
        for i_agents in range(2, len(agents)):  # 1 will never be "many"
            # -- aggregate the similar-believes (e.g., get the "worst best" in the standard case)
            top_i_sim_beliefs = similar_and_believes_sorted[:i_agents]
            agg_similar_believes = 1.0
            for sim_belief in top_i_sim_beliefs:
                agg_similar_believes = self.norm(agg_similar_believes, sim_belief)
            # -- determine if i is many
            i_is_many = self.conorm(
                i_agents_is_many(i_agents),
                i_agents_is_large_fraction(
                    i_agents, sum([ex > 0.5 for ex in expected])
                ),
            )
            # -- combine
            many_agents_believe_and_similar = self.conorm(
                many_agents_believe_and_similar,
                self.norm(agg_similar_believes, i_is_many),
            )

        # ---- aggregate cluster similar scores
        cluster_state_is_trusted = self.norm(
            a_trusted_has_similar, many_agents_believe_and_similar
        )

        # combine existence and similarity
        score = self.norm(cluster_exists, cluster_state_is_trusted)
        return score


def i_agents_is_many(i, k=3, lam=5):
    """Is the number of i agents large?

    Modeled with a Weibull CDF
    """
    return distribution.Weibull(k=k, lam=lam).cdf(i)


def i_agents_is_large_fraction(i, n, alpha=4, beta=2):
    """is the number i a large fraction of n expected agents?

    Modeled as a beta distribution CDF
    """
    return distribution.Beta(alpha=alpha, beta=beta).cdf(min(1.0, i / n))


def agent_high_track_score(agent, cluster):
    """if agent has a track in the cluster, return score"""
    if agent.ID in cluster.agent_IDs:
        tracks = cluster.get_tracks_by_agent_ID(agent.ID)
        # if len(tracks) != 1:
        #     logging.warning("Can only handle when a cluster has 1 track from an agent for now")
        return high_track_score(tracks[0].score)
    else:
        return 0.0


def high_track_score(s):
    """Probability computed directly from track score"""
    return math.exp(s) / (1 + math.exp(s))


def agent_expected_to_see(agent, cluster):
    """Observation model of expecting to be able to view object"""
    return float(agent.ID in cluster.agent_IDs)  # TODO: implement this...


def agent_is_trusted(agent):
    """Probability computed directly from agent"""
    return agent.trust


def agent_has_similar_track_state(agent, fused_state, cluster):
    """Computing similarity of track states"""
    if agent.ID in cluster.agent_IDs:
        # get track parameters
        tracks = cluster.get_tracks_by_agent_ID(agent.ID)
        # if len(tracks) != 1:
        #     logging.warning("Can only handle when a cluster has 1 track from an agent for now")
        a_track = tracks[0]
        x_a, P_a = a_track.x, a_track.P
        x_c, P_c = fused_state.x, fused_state.P

        # compute track similarities
        y = x_a - x_c
        S = P_a + P_c
        g = y.T @ np.linalg.solve(S, y)
        return 1 - distribution.Chi2(df=len(y)).cdf(g)
    else:
        return 0.0
