from functools import partial

import numpy as np

from mate.distribution import Beta
from mate.estimate import (
    DirectLinearKalmanTrustEstimator,
    MaximumLikelihoodTrustEstimator,
    SIRParticleFilterTrustEstimator,
    VotingTrustEstimator,
)
from mate.measurement import UncertainTrustArray


def run_estimator(estim, sampler, dt, n_frames, t0=0.0):
    t = t0
    for _ in range(n_frames):
        trust = sampler(t)
        estim.update(t, trust)
        t += dt
    return estim


def beta_sampler(beta, *args):
    return beta.rvs(1)


def boolean_beta_sampler(beta, *args):
    return beta.rvs(1) > 0.5


def set_up_linear_estimator(n_agents=6, t_prior=0.5, p_prior=0.3**2):
    estimator = DirectLinearKalmanTrustEstimator(
        t0=0, process_noise_diag=0.10, verbose=False
    )
    for j in range(n_agents):
        estimator.add_agent(ID=j, t_prior=t_prior, p_prior=p_prior)
    return estimator


#############################################################
# VECTOR/MATRIX IMPLEMENTATIONS
#############################################################


def test_direct_linear_kalman_add_agent():
    estimator = DirectLinearKalmanTrustEstimator(
        t0=0, process_noise_diag=0.10, verbose=False
    )
    assert estimator.n_agents == 0
    estimator.add_agent(ID=8, t_prior=0.5, p_prior=0.3**2)
    assert estimator.n_agents == 1
    estimator.add_agent(ID=2, t_prior=0.4, p_prior=0.4**2)
    assert estimator.n_agents == 2
    assert estimator.agent_ID_to_index == {8: 0, 2: 1}
    assert np.allclose(estimator.x, np.array([0.5, 0.4]))
    assert np.allclose(estimator.P, np.diag([0.3**2, 0.4**2]))


def test_direct_linear_kalman_run_no_correlations(dt=0.5, n_steps=20, n_agents=6):
    estimator = set_up_linear_estimator(n_agents=n_agents)
    prior = 0.5 * np.ones((n_agents,))
    agent_array = (np.arange(n_agents) + 1) / (n_agents + 1)
    for i in range(n_steps):
        t = i * dt
        trusts = np.clip(agent_array + 0.1 * np.random.rand(n_agents), 0, 1)
        confidence = np.diag(np.clip(0.7 + 0.1 * np.random.rand(n_agents), 0, 1))
        ID = np.arange(n_agents)
        trust = UncertainTrustArray(
            trust=trusts, confidence=confidence, prior=prior, ID=ID
        )
        estimator.update(t, trust)
    assert np.all(estimator.x < 1.0) and np.all(estimator.x > 0)
    assert np.all(abs(agent_array - estimator.x) < 0.1)
    assert np.all(estimator.P == np.diag(np.diag(estimator.P)))


def test_direct_linear_kalman_run_with_correlations(dt=0.5, n_steps=20, n_agents=6):
    estimator = set_up_linear_estimator(n_agents=n_agents)
    prior = 0.5 * np.ones((n_agents,))
    agent_array = (np.arange(n_agents) + 1) / (n_agents + 1)
    for i in range(n_steps):
        t = i * dt
        trusts = np.clip(agent_array + 0.1 * np.random.rand(n_agents), 0, 1)
        confidence = np.diag(np.clip(0.7 + 0.1 * np.random.rand(n_agents), 0, 1))
        confidence += 0.1 * np.random.rand(n_agents, n_agents)  # random correlations
        confidence = (confidence + confidence.T) / 2
        ID = np.arange(n_agents)
        trust = UncertainTrustArray(
            trust=trusts, confidence=confidence, prior=prior, ID=ID
        )
        estimator.update(t, trust)
    assert np.all(estimator.x < 1.0) and np.all(estimator.x > 0)
    assert np.all(abs(agent_array - estimator.x) < 0.1)
    assert np.any(estimator.P != np.diag(np.diag(estimator.P)))


#############################################################
# FLOAT IMPLEMENTATIONS
#############################################################


def test_voting_above():
    estim = VotingTrustEstimator(t0=0.0, p0=0.9, time_window=None)
    dt = 1.0 / 20
    n_frames = 100
    mean_target = 0.98
    var_target = 0.10**2
    sampler = partial(boolean_beta_sampler, Beta(mean=mean_target, var=var_target))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames)
    result = estim.binomial_test()
    assert result.pvalue > 0.05
    assert abs(result.statistic - mean_target) < 0.1


def test_voting_below():
    estim = VotingTrustEstimator(t0=0.0, p0=0.9, time_window=None)
    dt = 1.0 / 20
    n_frames = 100
    mean_target = 0.5
    var_target = 0.10**2
    sampler = partial(boolean_beta_sampler, Beta(mean=mean_target, var=var_target))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames)
    result = estim.binomial_test()
    assert result.pvalue < 0.05
    assert abs(result.statistic - mean_target) < 0.1


def test_mle_with_beta_unbounded():
    alpha0 = 1.0
    beta0 = 1.0
    dist = {"type": "Beta", "alpha": alpha0, "beta": beta0}
    estim = MaximumLikelihoodTrustEstimator(
        distribution=dist,
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
    var_target = 0.10**2
    sampler = partial(beta_sampler, Beta(mean=mean_target, var=var_target))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames)
    assert abs(estim.dist.mean - mean_target) < 0.1
    assert abs(estim.dist.var - var_target) < 0.1


def test_mle_with_beta_bounded_switch():
    alpha0 = 1.0
    beta0 = 1.0
    dist = {"type": "Beta", "alpha": alpha0, "beta": beta0}
    estim = MaximumLikelihoodTrustEstimator(
        distribution=dist,
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
    var_target_1 = 0.10**2
    sampler = partial(beta_sampler, Beta(mean=mean_target_1, var=var_target_1))
    estim = run_estimator(estim, sampler, dt=dt, n_frames=n_frames, t0=0.0)
    assert abs(estim.dist.mean - mean_target_1) < 0.1
    assert abs(estim.dist.var - var_target_1) < 0.1

    # second set
    n_frames = 20 * 6
    mean_target_2 = 0.4
    var_target_2 = 0.1**2
    sampler = partial(beta_sampler, Beta(mean=mean_target_2, var=var_target_2))
    estim = run_estimator(
        estim, sampler, dt=dt, n_frames=n_frames, t0=estim.t_last_update + dt
    )
    assert abs(estim.dist.mean - mean_target_2) < 0.1
    assert abs(estim.dist.var - var_target_2) < 0.1


def test_particle_filter_init():
    n_particles = 100
    prior = {"type": "Uniform", "a": 0.0, "b": 1.0}
    likelihood = {"type": "Normal", "mean": 0.0, "variance": 0.2**2}
    estim = SIRParticleFilterTrustEstimator(
        n_particles=n_particles, prior=prior, likelihood=likelihood
    )
    assert estim.n_particles == n_particles
    assert estim.weights[0] == 1 / n_particles


def test_particle_filter_propagate():
    n_particles = 10000
    prior = {"type": "Beta", "mean": 0.5, "var": 0.1**2}
    likelihood = {"type": "Normal", "mean": 0.0, "variance": 0.2**2}
    estim = SIRParticleFilterTrustEstimator(
        n_particles=n_particles, prior=prior, likelihood=likelihood
    )
    estim.propagate(timestamp=1.0)
    assert estim.variance - prior["var"] > 0.05
    assert abs(estim.mean - prior["mean"]) < 0.1


def test_particle_filter_update():
    n_particles = 10000
    prior = {"type": "Beta", "mean": 0.5, "var": 0.1**2}
    likelihood = {"type": "Normal", "mean": 0.0, "variance": 0.2**2}
    estim = SIRParticleFilterTrustEstimator(
        n_particles=n_particles, prior=prior, likelihood=likelihood
    )
    trust = 0.8
    t1 = 1.0
    assert np.isclose(estim.n_eff, n_particles)
    estim.update(timestamp=t1, trust=trust)
    assert estim.mean - prior["mean"] > 0.2
    assert (n_particles - estim.n_eff) > 500
