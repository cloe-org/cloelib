import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.observables.clusters.selection_function import SelectionFunction


def _test_selectionfunction(SF):
    z_test = np.linspace(0.01, 1.0, 20)
    zob_test = np.linspace(0.1, 1.1, 20)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 20)
    lob_test = np.logspace(0.2, 2.2, 20)

    print("    lnlambda")
    assert_allclose(SF.lnlambda(z_test, M_test), XXX)
    print("    scatter_lnl")
    assert_allclose(SF.scatter_lnl(z_test, M_test), XXX)
    print("    P_lnlbd")
    assert_allclose(SF.P_lnlbd(z_test, M_test, l_test), XXX)
    print("    scatter_lbdobs_lbd")
    assert_allclose(SF.scatter_lbdobs_lbd(z_test, l_test), XXX)
    print("    P_lbdobs_lbd")
    assert_allclose(SF.P_lbdobs_lbd(z_test, l_test, lob_test), XXX)
    print("    scatter_zobs_z")
    assert_allclose(SF.scatter_zobs_z(lob_test, z_test), XXX)
    print("    P_zobs_z")
    assert_allclose(SF.P_zobs_z(zob_test, lob_test, z_test), XXX)


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
    SF = SelectionFunction(**_sel_pars)
    _test_selectionfunction(SF)
