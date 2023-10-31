from typing import Any

from mate import connectives


def agent_score():
    pass


class ClusterScorer:
    def __init__(self, connective: connectives.StandardFuzzy) -> None:
        self.connective = connective

    def norm(self, a, b):
        return self.connective.norm(a, b)

    def conorm(self, a, b):
        return self.connective.conorm(a, b)

    def negation(self, a):
        return self.connective.negation(a)

    def __call__(self, cluster, agents, observations, *args: Any, **kwds: Any) -> Any:
        """Obtain a trust score for the cluster of interest

        trust represents the probability that the object is real and that the state is trusted
        """

        trusted = [agent_is_trusted(agent) for agent in agents]
        expected = [agent_expected_to_see(agent, cluster) for agent in agents]
        scores = [agent_high_track_score(agent, cluster) for agent in agents]
        beliefs = [self.norm(expected[i], scores[i]) for i in range(len(agents))]
        trusted_and_believes = [
            self.norm(trusted[i], beliefs[i]) for i in range(len(agents))
        ]
        similars = [agent_has_similar_track_state(agent, cluster) for agent in agents]
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
        for i in enumerate(agents):
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
                i_agents_is_large_fraction(i_agents, expected),
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
        for i, agent in enumerate(agents):
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
                i_agents_is_large_fraction(i_agents, expected),
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


def i_agents_is_many():
    raise


def i_agents_is_large_fraction():
    raise


def agent_high_track_score():
    raise


def agent_expected_to_see():
    raise


def agent_is_trusted():
    raise


def agent_has_similar_track_state():
    raise
