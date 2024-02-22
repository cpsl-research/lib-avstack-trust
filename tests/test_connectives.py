import numpy as np

from mate import connectives


def test_uncertaintrust():
    t, c, f = 0.5, 0.5, 0.5
    trust = connectives.UncertainTrustFloat(t, c, f, ID=0)
    t, c, f = 1.5, 0.5, 0.5
    try:
        trust = connectives.UncertainTrustFloat(t, c, f, ID=1)
    except ValueError:
        pass
    else:
        raise RuntimeError


def test_crisp_operator():
    op = connectives.CrispOperator
    assert op.conjunction(True, False) == 0
    assert op.disjunction(True, False) == 1


def test_zadeh_operator():
    op = connectives.ZadehOperator
    assert op.conjunction(0.5, 0.4) == 0.4
    assert op.disjunction(0.7, 0.2) == 0.7


def test_prob_operator():
    op = connectives.ProbabilityOperator
    assert np.isclose(op.conjunction(0.5, 0.4), 0.2)
    assert np.isclose(op.disjunction(0.7, 0.2), 0.76)


def test_einstein_operator():
    op = connectives.EinsteinOperator
    assert np.isclose(op.conjunction(0.5, 0.5), 0.2)
    assert np.isclose(op.disjunction(0.5, 0.5), 0.8)


def test_boundary_operator():
    op = connectives.BoundaryOperators
    assert np.isclose(op.conjunction(0.5, 0.6), 0.1)
    assert np.isclose(op.disjunction(0.7, 0.5), 1.0)


def test_certaintrust_operator():
    op = connectives.CertainTrustOperators
    a = connectives.UncertainTrustFloat(0.8, 0.9, 0.5, ID=0)
    b = connectives.UncertainTrustFloat(0.2, 0.1, 0.4, ID=1)
    c = op.conjunction(a, b)
    assert abs(a.t - c.t) > abs(b.t - c.t)
    assert b.c < c.c < a.c
    c = op.disjunction(a, b)
    assert abs(a.t - c.t) < abs(b.t - c.t)
    assert b.c < c.c < a.c
