"""
Tests for `AngularTwoPoint.get_pseudo_Cl`, i.e. the multiplication of the
theory Cls by the mixing matrices.

Each test uses tiny, hand-written numbers so the expected result can be
checked with pen and paper. We do not compute real Cls: `get_Cl` is
replaced by a fake that returns the numbers we choose, so the only thing
being tested is the mixing-matrix multiplication.

What `get_pseudo_Cl` should do (euclidlib internal format):
    - POS-POS: pseudo_Cl = M @ Cl
    - POS-SHE: the same matrix M multiplies the E and the B part separately.
    - SHE-SHE: the mixing matrix has three pieces, M_EE, M_BB and M_EB:
          EE_out = M_EE @ EE + M_BB @ BB
          BB_out = M_BB @ EE + M_EE @ BB
          EB_out = M_EB @ EB
          BE_out = M_EB @ BE

In every test the Cls are given at 3 multipoles (ell = 0, 1, 2) and the
mixing matrices turn them into 2 output values, so each matrix has
2 rows and 3 columns.
"""

import numpy as np
import pytest
from cosmolib.data import AngularPowerSpectrum

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

# A simple 2x3 mixing matrix: each output is the sum of two neighbouring ells.
#   output[0] = Cl[0] + Cl[1]
#   output[1] = Cl[1] + Cl[2]
SUM_NEIGHBOURS = np.array(
    [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Small helpers to set up the calculation
# ---------------------------------------------------------------------------


def make_tracer(tracer_class, n_bins=1):
    """Create an empty tracer of the given type.

    `get_pseudo_Cl` only needs to know the tracer type (positions or shear)
    and the number of redshift bins, so nothing else is set.
    """
    tracer = object.__new__(tracer_class)
    tracer.n_z_bins = n_bins
    return tracer


def wrap(array):
    """Put a plain array into the container used by cloelib/cosmolib."""
    return AngularPowerSpectrum(
        array=np.array(array, dtype=float),
        ell=np.array([10.0, 20.0]),
        lower=np.array([5.0, 15.0]),
        upper=np.array([15.0, 25.0]),
    )


def compute_pseudo_cl(tracer1, tracer2, fake_cls, mixing_matrices):
    """Run `get_pseudo_Cl`, using `fake_cls` instead of real Cls."""
    calculator = AngularTwoPoint(tracer1, tracer2)
    calculator.get_Cl = lambda ells, nl, ks: {
        key: wrap(cl) for key, cl in fake_cls.items()
    }
    mixing_matrices = {key: wrap(m) for key, m in mixing_matrices.items()}
    return calculator.get_pseudo_Cl(nl=None, ks=None, mixing_matrix=mixing_matrices)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pos_pos():
    key = ("POS", "POS", 1, 1)
    cl = [1.0, 2.0, 3.0]

    result = compute_pseudo_cl(
        make_tracer(PositionsTracer),
        make_tracer(PositionsTracer),
        fake_cls={key: cl},
        mixing_matrices={key: SUM_NEIGHBOURS},
    )

    # [1 + 2, 2 + 3]
    np.testing.assert_allclose(result[key].array, [3.0, 5.0])


@pytest.mark.parametrize(
    "tracer1, tracer2",
    [(PositionsTracer, ShearTracer), (ShearTracer, PositionsTracer)],
    ids=["POS-SHE", "SHE-POS"],
)
def test_pos_she(tracer1, tracer2):
    key = ("POS", "SHE", 1, 1)
    E = [1.0, 2.0, 3.0]
    B = [10.0, 20.0, 30.0]

    result = compute_pseudo_cl(
        make_tracer(tracer1),
        make_tracer(tracer2),
        fake_cls={key: [E, B]},
        mixing_matrices={key: SUM_NEIGHBOURS},
    )

    E_out, B_out = result[key].array
    np.testing.assert_allclose(E_out, [3.0, 5.0])  # [1 + 2, 2 + 3]
    np.testing.assert_allclose(B_out, [30.0, 50.0])  # [10 + 20, 20 + 30]


@pytest.mark.parametrize(
    "tracer1, n_bins1, tracer2, n_bins2",
    [
        (PositionsTracer, 2, ShearTracer, 3),
        (ShearTracer, 3, PositionsTracer, 2),
    ],
    ids=["POS-SHE", "SHE-POS"],
)
def test_pos_she_different_number_of_bins(tracer1, n_bins1, tracer2, n_bins2):
    # 2 POS bins and 3 SHE bins give 2 x 3 = 6 pairs, always named
    # ("POS", "SHE", POS bin, SHE bin), whatever the order of the tracers.
    # Each pair gets the same Cl but a different multiple of SUM_NEIGHBOURS,
    # so we can check that every pair is computed with its own matrix.
    E = [1.0, 2.0, 3.0]
    B = [10.0, 20.0, 30.0]
    factor = {
        (1, 1): 1, (1, 2): 2, (1, 3): 3,
        (2, 1): 4, (2, 2): 5, (2, 3): 6,
    }  # fmt: skip
    fake_cls = {("POS", "SHE", i, j): [E, B] for (i, j) in factor}
    mixing_matrices = {
        ("POS", "SHE", i, j): f * SUM_NEIGHBOURS for (i, j), f in factor.items()
    }

    result = compute_pseudo_cl(
        make_tracer(tracer1, n_bins1),
        make_tracer(tracer2, n_bins2),
        fake_cls,
        mixing_matrices,
    )

    # Exactly the 6 pairs, no pair missing and none extra.
    assert set(result) == set(fake_cls)
    for (i, j), f in factor.items():
        E_out, B_out = result[("POS", "SHE", i, j)].array
        np.testing.assert_allclose(E_out, [f * 3.0, f * 5.0])
        np.testing.assert_allclose(B_out, [f * 30.0, f * 50.0])


# SHE-SHE: one test per output component (EE, BB, EB, BE), so a failing test
# tells you directly which multiplication is wrong.
#
# The numbers are chosen so that ANY mistake gives a different answer:
#   - each input component has a very different size (1s, 10s, 100s, 1000s),
#     so using the wrong component (e.g. BB instead of EE) is visible;
#   - each mixing matrix picks different ells, so using the wrong matrix
#     (e.g. M_EE instead of M_BB) is visible too.
# Every other combination of one or two (matrix @ component) terms gives a
# result different from the correct one.
EE = [1.0, 2.0, 3.0]
BB = [10.0, 20.0, 30.0]
EB = [100.0, 200.0, 300.0]
BE = [1000.0, 2000.0, 3000.0]

M_EE = SUM_NEIGHBOURS  # output = [Cl[0] + Cl[1], Cl[1] + Cl[2]]
M_BB = [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]  # output = [2 * Cl[0], 2 * Cl[2]]
M_EB = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # output = [Cl[1], Cl[2]]


def she_she_pseudo_cl():
    """Run `get_pseudo_Cl` for SHE-SHE with the numbers above.

    Returns the output as a 2x2 grid: [[EE, EB], [BE, BB]].
    """
    key = ("SHE", "SHE", 1, 1)
    result = compute_pseudo_cl(
        make_tracer(ShearTracer),
        make_tracer(ShearTracer),
        fake_cls={key: [[EE, EB], [BE, BB]]},
        mixing_matrices={key: [M_EE, M_BB, M_EB]},
    )
    return result[key].array


def test_she_she_EE():
    # EE_out = M_EE @ EE + M_BB @ BB
    #        = [1 + 2, 2 + 3] + [2 * 10, 2 * 30]
    EE_out = she_she_pseudo_cl()[0, 0]
    np.testing.assert_allclose(EE_out, [23.0, 65.0])


def test_she_she_BB():
    # BB_out = M_BB @ EE + M_EE @ BB
    #        = [2 * 1, 2 * 3] + [10 + 20, 20 + 30]
    BB_out = she_she_pseudo_cl()[1, 1]
    np.testing.assert_allclose(BB_out, [32.0, 56.0])


def test_she_she_EB():
    # EB_out = M_EB @ EB = [200, 300]
    EB_out = she_she_pseudo_cl()[0, 1]
    np.testing.assert_allclose(EB_out, [200.0, 300.0])


def test_she_she_BE():
    # BE_out = M_EB @ BE = [2000, 3000]
    BE_out = she_she_pseudo_cl()[1, 0]
    np.testing.assert_allclose(BE_out, [2000.0, 3000.0])


def test_each_bin_pair_uses_its_own_mixing_matrix():
    # Two redshift bins give three pairs: (1, 1), (1, 2) and (2, 2).
    # All pairs get the same Cl, but a different mixing matrix
    # (SUM_NEIGHBOURS times 1, 2 and 3), so each result should be different.
    cl = [1.0, 2.0, 3.0]
    fake_cls = {
        ("POS", "POS", 1, 1): cl,
        ("POS", "POS", 1, 2): cl,
        ("POS", "POS", 2, 2): cl,
    }
    mixing_matrices = {
        ("POS", "POS", 1, 1): 1 * SUM_NEIGHBOURS,
        ("POS", "POS", 1, 2): 2 * SUM_NEIGHBOURS,
        ("POS", "POS", 2, 2): 3 * SUM_NEIGHBOURS,
    }

    result = compute_pseudo_cl(
        make_tracer(PositionsTracer, n_bins=2),
        make_tracer(PositionsTracer, n_bins=2),
        fake_cls,
        mixing_matrices,
    )

    np.testing.assert_allclose(result[("POS", "POS", 1, 1)].array, [3.0, 5.0])
    np.testing.assert_allclose(result[("POS", "POS", 1, 2)].array, [6.0, 10.0])
    np.testing.assert_allclose(result[("POS", "POS", 2, 2)].array, [9.0, 15.0])


def test_output_keeps_the_mixing_matrix_ells():
    # The pseudo-Cl is given at the output multipoles of the mixing matrix,
    # so the ell values and bin edges must be copied from it.
    key = ("POS", "POS", 1, 1)

    result = compute_pseudo_cl(
        make_tracer(PositionsTracer),
        make_tracer(PositionsTracer),
        fake_cls={key: [1.0, 2.0, 3.0]},
        mixing_matrices={key: SUM_NEIGHBOURS},
    )

    np.testing.assert_array_equal(result[key].ell, [10.0, 20.0])
    np.testing.assert_array_equal(result[key].lower, [5.0, 15.0])
    np.testing.assert_array_equal(result[key].upper, [15.0, 25.0])


def test_unsupported_tracers_raise_an_error():
    calculator = AngularTwoPoint(object(), object())
    with pytest.raises(ValueError, match="Unsupported tracer pair"):
        calculator.get_pseudo_Cl(nl=None, ks=None, mixing_matrix={})
