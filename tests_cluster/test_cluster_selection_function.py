# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.observables.clusters.selection_function import GaussianSelectionFunction


def _test_selectionfunction(SF):
    z_test = np.linspace(0.01, 1.0, 5)
    zob_test = np.linspace(0.1, 1.1, 5)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 5)
    lob_test = np.logspace(0.2, 2.2, 5)

    XXX = 1

    print("    lnlambda")
    _lnlambda_ref = [-1.533121, -1.423534, -1.3337, -1.257575, -1.191523]
    assert_allclose(
        SF.mass_lambda_true.lnlambda(z_test, M_test)[:, 0], _lnlambda_ref, rtol=1e-05
    )
    print("    scatter_lnl")
    assert_allclose(SF.mass_lambda_true.scatter_lnl(z_test, M_test), 0.1, rtol=1e-05)
    print("    prob_true_richness_given_mass")
    assert_allclose(
        SF.mass_lambda_true.prob_true_richness_given_mass(z_test, M_test, l_test),
        0,
        atol=1e-10,
        rtol=1e-05,
    )
    print("    scatter_lbdobs_lbd")
    _scatter_lbdobs_lbd_ref = [0.101, 0.113324, 0.127151, 0.142666, 0.160074]
    assert_allclose(
        SF.scatter_lbdobs_lbd(z_test, l_test)[0], _scatter_lbdobs_lbd_ref, rtol=1e-05
    )
    print("    P_lbdobs_lbd")
    _P_lbdobs_lbd_ref = [2.06231e-07, 0.00000e00, 0.00000e00, 0.00000e00, 0.00000e00]
    assert_allclose(
        SF.P_lbdobs_lbd(z_test, l_test, lob_test)[0, 0], _P_lbdobs_lbd_ref, rtol=1e-05
    )
    print("    scatter_zobs_z")
    _scatter_zobs_z_ref = [0.159489, 0.526937, 1.635393, 5.087122, 15.948932]
    assert_allclose(SF.scatter_zobs_z(lob_test, z_test), _scatter_zobs_z_ref, rtol=1e-5)
    print("    P_zobs_z")
    _P_zobs_z_ref = [
        2.133197e00,
        7.240213e-01,
        2.365759e-01,
        7.777955e-02,
        2.497394e-02,
    ]
    assert_allclose(
        SF.P_zobs_z(zob_test, lob_test, z_test)[0], _P_zobs_z_ref, rtol=5e-07
    )


def test_selectionfunction():
    # SelectionFunction
    print("# SelectionFunction")
    _sel_pars = dict(
        A_l=0.5,
        B_l=0.6,
        C_l=0.5,
        sig_A_l=0.1,
        sig_B_l=0.0,
        sig_C_l=0.0,
        sig_lambda_norm=0.1,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.1,
        sig_z_z=0.1,
        sig_z_lambda=0.1,
    )
    SF = GaussianSelectionFunction(**_sel_pars)
    _test_selectionfunction(SF)
