from functools import partial

from mate.distribution import Beta
from mate.trust.estimate import MaximumLikelihoodTrustEstimator


def run_estimator(estim, sampler, dt, n_frames, t0=0.0):
    t = t0
    for i in range(n_frames):
        trust = sampler(t)
        estim.update(t, trust)
        t += dt
    return estim


def beta_sampler(beta, *args):
    return beta.rvs(1)


def test_mle_with_beta_unbounded():
    alpha0 = 1.0
    beta0 = 1.0
    dist = Beta(alpha=alpha0, beta=beta0)
    estim = MaximumLikelihoodTrustEstimator(
        dist=dist,
        update_rate=10,
        time_window=None,
        forgetting=0.0,
        max_variance=None,
        n_min_update=2,
        update_weighting="uniform",
    )
    dt = 1.0 / 20
    n_frames = 100
    mean_target = 0.7
    var_target = 0.10
    sampler = partial(beta_sampler, Beta(mean=mean_target, var=var_target))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames)
    assert abs(estim.dist.mean - mean_target) < 0.1
    assert abs(estim.dist.var - var_target) < 0.1


def test_mle_with_beta_bounded_switch():
    alpha0 = 1.0
    beta0 = 1.0
    dist = Beta(alpha=alpha0, beta=beta0)
    estim = MaximumLikelihoodTrustEstimator(
        dist=dist,
        update_rate=10,
        time_window=5,
        forgetting=0.0,
        max_variance=None,
        n_min_update=2,
        update_weighting="uniform",
    )
    dt = 1.0 / 20

    # first set
    n_frames = 20 * 4
    mean_target_1 = 0.7
    var_target_1 = 0.10
    sampler = partial(beta_sampler, Beta(mean=mean_target_1, var=var_target_1))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames, t0=0.0)
    assert abs(estim.dist.mean - mean_target_1) < 0.1
    assert abs(estim.dist.var - var_target_1) < 0.1

    # second set
    n_frames = 20 * 6
    mean_target_2 = 0.4
    var_target_2 = 0.1
    sampler = partial(beta_sampler, Beta(mean=mean_target_2, var=var_target_2))
    estim = run_estimator(
        estim, sampler, dt=dt, n_frames=n_frames, t0=estim.t_last_update + dt
    )
    assert abs(estim.dist.mean - mean_target_2) < 0.1
    assert abs(estim.dist.var - var_target_2) < 0.1
