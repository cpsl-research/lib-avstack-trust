import numpy as np

from mate import sampling


def test_all_samplings_on_n_weights():
    n = 100
    algs = [
        sampling.multinomial_sampling,
        sampling.residual_sampling,
        sampling.stratified_sampling,
        sampling.systematic_sampling,
    ]
    weights = np.random.rand(n)
    for alg in algs:
        indices = alg(weights, n=n)
