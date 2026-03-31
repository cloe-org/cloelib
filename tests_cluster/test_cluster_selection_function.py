# import jax.numpy as np

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises

from cloelib.observables.clusters.halo_mass_observable import (
    LognormalPowerLawHaloMassObservable,
)
from cloelib.observables.clusters.selection_function import (
    GaussianSelectionFunction,
    NumericalSelectionFunction,
)


def _get_test_gaussian_sf():
    _lambda_true_dist_pars = dict(
        A_l=0.5,
        B_l=0.6,
        C_l=0.5,
        sig_A_l=0.1,
        sig_B_l=0.0,
        sig_C_l=0.0,
    )
    _sel_pars = dict(
        sig_lambda_norm=0.1,
        sig_lambda_z=0.1,
        sig_lambda_exponent=0.1,
        sig_z_z=0.1,
        sig_z_lambda=0.1,
    )
    return GaussianSelectionFunction(
        **_sel_pars,
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            **_lambda_true_dist_pars
        ),
        lambda_tab_integ=[31, 31, 31, 51],
        z_tab_integ=31,
    )


def test_gaussian_selectionfunction():
    print("# SelectionFunction")
    selection_function = _get_test_gaussian_sf()

    # tests
    z_test = np.linspace(0.01, 1.0, 5)
    zob_test = np.linspace(0.1, 1.1, 5)
    M_test = 1.0e14
    l_test = np.logspace(0.0, 2.0, 5)
    lob_test = np.logspace(0.2, 2.2, 5)

    # Tests

    print("    lnrichness")
    _lnrichness_ref = [-1.533121, -1.423534, -1.3337, -1.257575, -1.191523]
    assert_allclose(
        selection_function.halo_mass_observable._mean_lnrichness(z_test, M_test),
        _lnrichness_ref,
        rtol=1e-05,
    )
    print("    scatter_lnrichness")
    assert_allclose(
        selection_function.halo_mass_observable.scatter_lnrichness(z_test, M_test),
        0.1,
        rtol=1e-05,
    )
    print("    prob_true_richness")
    assert_allclose(
        selection_function.halo_mass_observable.pdf_richness(z_test, M_test, l_test),
        0,
        atol=1e-10,
        rtol=1e-05,
    )
    print("    scatter_lambda_obs")
    _scatter_lambda_obs_ref = [0.101, 0.113324, 0.127151, 0.142666, 0.160074]
    assert_allclose(
        selection_function._scatter_lambda_obs(z_test[:, None], l_test[None, :])[0],
        _scatter_lambda_obs_ref,
        rtol=1e-05,
    )
    print("    scatter_z_obs")
    _scatter_z_obs_ref = [0.159489, 0.526937, 1.635393, 5.087122, 15.948932]
    assert_allclose(
        selection_function.scatter_z_obs(lob_test, z_test),
        _scatter_z_obs_ref,
        rtol=1e-5,
    )

    # Prob functions tests
    print("    prob_lambda_obs")
    _prob_lambda_obs_ref = [2.06231e-07, 0.00000e00, 0.00000e00, 0.00000e00, 0.00000e00]
    assert_allclose(
        selection_function._prob_lambda_obs(
            z_test[:, None, None], l_test[None, :, None], lob_test[None, None, :]
        )[0, 0],
        _prob_lambda_obs_ref,
        rtol=1e-05,
    )
    print("    prob_z_obs")
    _prob_z_obs_ref = [
        2.133197e00,
        7.240213e-01,
        2.365759e-01,
        7.777955e-02,
        2.497394e-02,
    ]
    assert_allclose(
        selection_function._prob_z_obs(
            zob_test[:, None, None], lob_test[None, :], z_test
        )[0, 0],
        _prob_z_obs_ref,
        rtol=5e-07,
    )


def _gen_gaussian_selcl_data(gaussian_sf, arrays):
    sel_cl_data = {"arrays": arrays}

    sel_cl_data["step_size"] = {
        "z_obs": (
            sel_cl_data["arrays"]["z_obs"][1:] - sel_cl_data["arrays"]["z_obs"][:-1]
        ).mean(),
        "lambda_obs": (
            sel_cl_data["arrays"]["lambda_obs"][1:]
            - sel_cl_data["arrays"]["lambda_obs"][:-1]
        ).mean(),
    }
    sel_cl_data["area_tile"] = np.ones(3)

    # tables
    _prob_func = lambda zob, lob, ztr, ltr, alpha: (
        gaussian_sf._prob_lambda_obs(
            ztr[None, :, None], ltr[None, None, :], lob[:, None, None]
        )[None, None, ...]
        * gaussian_sf._prob_z_obs(zob[:, None, None], lob[None, :, None], ztr)[
            None, :, :, :, None
        ]
        * alpha[:, None, None, None, None]
    )

    sel_cl_data["tables"] = {
        "prob_lambda_z_obs": _prob_func(
            sel_cl_data["arrays"]["z_obs"],
            sel_cl_data["arrays"]["lambda_obs"],
            sel_cl_data["arrays"]["z_true"],
            sel_cl_data["arrays"]["lambda_true"],
            sel_cl_data["area_tile"],
        ),
        "completeness": np.ones(
            (
                sel_cl_data["area_tile"].size,
                sel_cl_data["arrays"]["z_true"].size,
                sel_cl_data["arrays"]["lambda_true"].size,
            )
        ),
        "purity": np.ones(
            (
                sel_cl_data["area_tile"].size,
                sel_cl_data["arrays"]["z_obs"].size,
                sel_cl_data["arrays"]["lambda_obs"].size,
            )
        ),
    }

    return sel_cl_data


def test_interpolated_selectionfunction_unittest():

    test_arrays = {
        "z_true": np.linspace(0, 3, 31),
        "lambda_true": np.linspace(5, 300, 29),
        "z_obs": np.linspace(0.3, 1.5, 25),
        "lambda_obs": np.linspace(25, 200, 27),
    }
    print("Test with gaussian input data")
    gaussian_sf = _get_test_gaussian_sf()
    sel_cl_data = _gen_gaussian_selcl_data(gaussian_sf, test_arrays)

    sfn = NumericalSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sel_cl_data=sel_cl_data,
        prob_contains_completeness=False,
        extrapolate=0,
    )

    # Prob functions tests
    interps = sfn._build_windows_interpolators(
        z_obs_edges=test_arrays["z_obs"][::3],
        lambda_obs_edges=test_arrays["lambda_obs"][::2],
    )

    # data generated by the code, must be updated at some point
    ref_data = np.array(
        [
            [9.778339e-05, 9.812106e-05, 9.844903e-05, 9.876724e-05],
            [9.997279e-05, 1.003186e-04, 1.006545e-04, 1.009805e-04],
            [9.997279e-05, 1.003186e-04, 1.006545e-04, 1.009805e-04],
        ]
    )
    x = np.linspace(2.99, 3.01, 3)
    y = np.linspace(36.5, 36.7, 4)
    assert_allclose(interps[1][1](x, y), ref_data, rtol=1e-06)


def test_interpolated_selectionfunction_compare_with_gauss():

    test_arrays = {
        "z_true": np.linspace(0, 3, 31),
        "lambda_true": np.linspace(5, 300, 29),
        "z_obs": np.linspace(0.3, 1.5, 25),
        "lambda_obs": np.linspace(25, 200, 27),
    }
    print("Test with gaussian input data")
    gaussian_sf = _get_test_gaussian_sf()
    sel_cl_data = _gen_gaussian_selcl_data(gaussian_sf, test_arrays)

    sfn = NumericalSelectionFunction(
        halo_mass_observable=LognormalPowerLawHaloMassObservable(
            A_l=52.0,
            B_l=0.9,
            C_l=0.5,
            sig_A_l=0.2,
            sig_B_l=-0.05,
            sig_C_l=0.001,
        ),
        sel_cl_data=sel_cl_data,
        prob_contains_completeness=False,
        extrapolate=0,
    )

    # the test below is not valid because:
    # 1. They are not exactly the same distributions

    gaussian_sf.lambda_tab_integ = [31]
    gaussian_sf.z_tab_integ = 151

    wf_g = gaussian_sf.window_redshift_richness_observed(
        z_obs_edges=test_arrays["z_obs"][[0, -1]],
        lambda_obs_edges=test_arrays["lambda_obs"][[0, -1]],
        z_true=np.linspace(0, 3, 31),
        mass=1e14 * np.ones(2),
        lambda_true=np.linspace(5, 300, 29),
    )
    wf_n = sfn.window_redshift_richness_observed(
        z_obs_edges=test_arrays["z_obs"][[0, -1]],
        lambda_obs_edges=test_arrays["lambda_obs"][[0, -1]],
        z_true=np.linspace(0, 3, 31),
        mass=1e14 * np.ones(2),
        lambda_true=np.linspace(5, 300, 29),
    )

    # compare zeros
    _zeros = wf_g == 0
    assert_allclose(wf_g[_zeros], wf_n[_zeros], atol=2e-2)
    # compare non zeros
    print(f"nz: {(~_zeros).sum():}")
    assert_allclose(wf_g[~_zeros], wf_n[~_zeros], rtol=1e-100)
