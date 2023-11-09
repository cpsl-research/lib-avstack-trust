_base_ = "../_base_/base_command_center.py"

models = dict(
    clustering=dict(
        type="ClusterTracker",
        clusterer=dict(type="SampledAssignmentClustering"),
        tracker=dict(type=""),
        platform="GlobalOrigin3D",
    ),
    fusion=dict(type="CovarianceIntersectionFusion"),
    trust=dict(
        type="PointBasedTrust",
        cluster_scorer=dict(type="ClusterScorer", connective="StandardFuzzy"),
        agent_scorer=dict(type="AgentScorer"),
        estimator=dict(
            type="MaximumLikelihoodTrustEstimator",
            distribution="Beta",
            alpha=None,
            beta=None,
            phi=0.5,
            lam=0.1,
        ),
    ),
)
