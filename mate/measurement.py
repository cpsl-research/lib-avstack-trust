import math
from collections import namedtuple
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
    from mate.connectives import _DeMorganTriple, StandardFuzzy

import numpy as np
from avstack.config import MODELS, ConfigDict
from avstack.geometry.datastructs import Pose
from avstack.modules.tracking.tracks import _TrackBase

from mate import distribution
from mate.fov import FieldOfView
from mate.state import Agent


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


#############################################################
# MEASUREMENT DATA STRUCTURES
#############################################################


def check_range(value):
    if not np.all((0 <= value) & (value <= 1)):
        raise ValueError("Inputs to uncertain trust must be on [0,1]")


def check_type_as_array(value):
    if not isinstance(value, np.ndarray):
        raise TypeError("Input must be an np array")


def check_type_as_float(value):
    if not isinstance(value, float):
        raise TypeError("Input must be a float")


class _Trust(namedtuple("Trust", "trust ID")):
    def __new__(cls, trust, ID):
        for a in (trust,):
            for check in cls.checks:
                check(a)

        self = super().__new__(cls, trust=trust, ID=ID)
        return self

    @property
    def t(self):
        return self.trust


class TrustArray(_Trust):
    """A vector of trust measurements without uncertainty"""

    checks = [check_range, check_type_as_array]

    def __iter__(self) -> Iterator:
        return zip(self.trust, self.ID)


class TrustFloat(_Trust):
    """A float of trust measurement with without uncertainty"""

    checks = [check_range, check_type_as_float]


class _UncertainTrust(namedtuple("UncertainTrust", "trust confidence prior ID")):
    def __new__(cls, trust, confidence, prior, ID):
        for a in (trust, confidence, prior):
            for check in cls.checks:
                check(a)

        self = super().__new__(
            cls, trust=trust, confidence=confidence, prior=prior, ID=ID
        )
        return self

    @property
    def t(self):
        return self.trust

    @property
    def c(self):
        return self.confidence

    @property
    def f(self):
        return self.prior


class UncertainTrustArray(_UncertainTrust):
    """A vector of trust measurements and a matrix of uncertainties/correlations"""

    checks = [check_range, check_type_as_array]

    def __iter__(self) -> Iterator:
        return zip(self.trust, self.confidence, self.prior, self.ID)


class UncertainTrustFloat(_UncertainTrust):
    """A float of trust measurement uncertainties w/o correlations"""

    checks = [check_range, check_type_as_float]


#############################################################
# MEASUREMENT GENERATION
#############################################################


class PseudoMeasurementBase:
    def __init__(self, connective: "StandardFuzzy") -> None:
        self.connective = MODELS.build(connective)

    def norm(self, a, b):
        return self.connective.norm(a, b)

    def conorm(self, a, b):
        return self.connective.conorm(a, b)

    def negation(self, a):
        return self.connective.negation(a)


@MODELS.register_module()
class AgentScorer(PseudoMeasurementBase):
    def __init__(
        self,
        connective: ConfigDict = {"type": "StandardFuzzy"},
        clusterer: ConfigDict = {
            "type": "SampledAssignmentClusterer",
            "assign_radius": 4,
        },
    ) -> None:
        super().__init__(connective)
        self.clusterer = MODELS.build(clusterer)

    def __call__(
        self,
        tracks: Dict[int, List[_TrackBase]],
        agents: Dict[int, Agent],
        poses: Dict[int, Pose],
        fovs: Dict[int, FieldOfView],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Obtain a trust pseudo-measurement for each agent's objects

        Inputs:
        - tracks: a dict mapping agent ID to local tracks
        - poses: a dict mapping agent ID to poses
        - fovs: a dict mapping agent ID to field of view models

        Outputs:
        - score: an estimate of trust on [0, 1] for each agent
        - uncertainty: an estimate of noise variance on (0, inf) for each agent

        this trust scorer is one-trust-per-agent
        assumes that if one track is compromised, the
        agent can no longer be trusted at all
        """
        # initialize
        n_agents = len(tracks)
        agent_IDs = list(tracks.keys())
        score = {}
        uncertainty = {}

        # get scores for each agent
        for agent_k_ID in agent_IDs:
            # loop over tracks and use other agent IDs
            score_tracks = []
            weight_tracks = []
            other_agent_IDs = [ID for ID in agent_IDs if ID != agent_k_ID]
            for track_j in enumerate(tracks[agent_k_ID]):
                track_j_ID = track_j.ID

                # compute scores
                score_q_is_large = np.array(
                    [q_is_large(q) for q in range(1, len(other_agent_IDs))]
                )
                score_agent_believes_track_consistent = np.array(
                    [
                        agent_believes_track_consistent(
                            track_j,
                            tracks[agent_k2_ID],
                            poses[agent_k2_ID],
                            fovs[agent_k2_ID],
                        )
                        for agent_k2_ID in other_agent_IDs
                    ]
                )
                score_agent_believes_track_consistent_sorted = sorted(
                    score_agent_believes_track_consistent
                )
                score_q_agents_believe = [
                    score_agent_believes_track_consistent_sorted[-q]
                    for q in range(1, len(other_agent_IDs))
                ]

                # compute weights
                w_trusted = np.array(
                    [
                        agent_is_trusted(agents[agent_k2_ID])
                        for agent_k2_ID in other_agent_IDs
                    ]
                )
                w_expected = np.array(
                    [
                        agent_expected_to_see(
                            track_j, poses[agent_k2_ID], fovs[agent_k2_ID]
                        )
                        for agent_k2_ID in other_agent_IDs
                    ]
                )
                w_expected_trusted = w_trusted * w_expected

                # ----------------------------------
                # TRACK CONSISTENT
                # ----------------------------------

                # -- not inconsistent with trusted
                scores_1 = self.norm(
                    score_agent_believes_track_consistent, weight=w_expected_trusted
                )

                # -- consistent with at least one trusted
                scores_2 = self.conorm(
                    score_agent_believes_track_consistent, weight=w_expected_trusted
                )

                # -- consistent with many untrusted
                scores_3 = self.conorm(
                    self.norm(score_q_is_large, score_q_agents_believe)
                )

                # combine scores
                track_j_consistent = self.conorm(
                    self.norm(scores_1, scores_2),
                    self.norm(scores_1, scores_3),
                )

                # ----------------------------------
                # TRACK LOW CONFIDENCE
                # ----------------------------------

                track_j_low_confidence = self.conorm(
                    track_is_young(track_j), track_is_low_score(track_j)
                )

                # ----------------------------------
                # COMBINE OUTCOMES FOR THIS TRACK
                # ----------------------------------

                score_tracks.append(
                    self.conorm(track_j_consistent, track_j_low_confidence)
                )
                weight_tracks.append(
                    0.0
                )  # TODO: add up the contribution of each track to the overall

            # ----------------------------------
            # COMBINE ALL TRACKS TOGETHER
            # ----------------------------------
            objs_consistent = self.norm(score_tracks, weight=weight_tracks)

        return score, uncertainty


@MODELS.register_module()
class AgentObjectLocalScorer(PseudoMeasurementBase):
    pass


@MODELS.register_module()
class AgentFreeSpaceScorer(PseudoMeasurementBase):
    def __call__(
        self,
        tracks: Dict[int, List[_TrackBase]],
        poses: Dict[int, Pose],
        fovs: Dict[int, FieldOfView],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Obtain a trust pseudo-measurement for each agent's free space

        Inputs:
        - tracks: a dict mapping agent ID to local tracks
        - poses: a dict mapping agent ID to poses
        - fovs: a dict mapping agent ID to field of view models

        Outputs:
        - score: an estimate of trust on [0, 1] for each fov subset overlap
        - uncertainty: an estimate of noise variance on (0, inf) for each fov subset
        """
        raise NotImplementedError


@MODELS.register_module()
class ClusterScorer(PseudoMeasurementBase):
    def __init__(self, connective: "_DeMorganTriple") -> None:
        super().__init__(connective=connective)

    def __call__(
        self, group_track, agents: Dict[int, Agent], *args: Any, **kwds: Any
    ) -> Any:
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


def track_is_young(track):
    raise NotImplementedError


def track_is_low_score(track):
    raise NotImplementedError


def agent_believes_track_consistent():
    raise NotImplementedError


def q_is_large(q):
    raise NotImplementedError


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
