"""Unit tests for cloelib.auxiliary.geometry and the curved photo-RSD kernel."""

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloelib.auxiliary.geometry import (
    cos_K,
    curvature,
    is_flat,
    limber_wavenumber,
    potential_response,
    radial_distance,
    transverse_distance,
)
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.observables.photo.positions import get_photo_rsd

H0 = 67.0
D_H = SPEED_OF_LIGHT / 1000 / H0
CHI = np.linspace(10.0, 9000.0, 60)
OMEGA_K = [-0.3, -0.05, -1e-6, 0.0, 1e-6, 0.05, 0.3]
ELLS = np.arange(2.0, 200.0, 7.0)


def _f_K(chi, Omega_k0):
    if Omega_k0 > 0:
        return D_H / np.sqrt(Omega_k0) * np.sinh(np.sqrt(Omega_k0) * chi / D_H)
    if Omega_k0 < 0:
        return D_H / np.sqrt(-Omega_k0) * np.sin(np.sqrt(-Omega_k0) * chi / D_H)
    return chi


@pytest.mark.parametrize("Omega_k0", OMEGA_K)
def test_transverse_distance_closed_form(Omega_k0):
    assert_allclose(
        transverse_distance(CHI, Omega_k0, H0), _f_K(CHI, Omega_k0), rtol=1e-12
    )


@pytest.mark.parametrize("Omega_k0", OMEGA_K)
def test_radial_distance_inverts_transverse(Omega_k0):
    d_M = transverse_distance(CHI, Omega_k0, H0)
    assert_allclose(radial_distance(d_M, Omega_k0, H0), CHI, rtol=1e-12)


@pytest.mark.parametrize("Omega_k0", OMEGA_K)
def test_cos_K_is_derivative_of_transverse_distance(Omega_k0):
    grad = jax.vmap(jax.grad(lambda chi: transverse_distance(chi, Omega_k0, H0)))
    assert_allclose(grad(CHI), cos_K(CHI, Omega_k0, H0), rtol=1e-10)


def test_transverse_distance_differentiable_through_flat():
    dfK_dOk = jax.jacfwd(lambda Ok: transverse_distance(CHI, Ok, H0))(0.0)
    assert_allclose(dfK_dOk, CHI**3 / (6 * D_H**2), rtol=1e-12)


def test_transverse_distance_odd():
    for Omega_k0 in (-0.3, 0.3):
        assert_allclose(
            transverse_distance(-CHI, Omega_k0, H0),
            -transverse_distance(CHI, Omega_k0, H0),
            rtol=1e-14,
        )


def test_is_flat():
    assert is_flat(0.0)
    assert not is_flat(0.01)
    seen = []
    jax.jit(lambda Ok: seen.append(is_flat(Ok)) or Ok)(0.0)
    assert seen == [False]


def test_limber_quantities_reduce_to_flat():
    f_K = CHI
    assert_allclose(limber_wavenumber(ELLS, f_K, 0.0), (ELLS[:, None] + 0.5) / f_K)
    assert_allclose(potential_response(ELLS, f_K, 0.0), np.ones((len(ELLS), len(f_K))))


@pytest.mark.parametrize("Omega_k0", [-0.3, 0.3])
def test_limber_quantities_curved(Omega_k0):
    f_K = transverse_distance(CHI, Omega_k0, H0)
    K = curvature(Omega_k0, H0)
    nu2 = (ELLS[:, None] + 0.5) ** 2
    assert_allclose(limber_wavenumber(ELLS, f_K, K) ** 2, nu2 / f_K**2 - K, rtol=1e-12)
    assert_allclose(
        potential_response(ELLS, f_K, K), nu2 / (nu2 - 4 * K * f_K**2), rtol=1e-12
    )


def _rsd_interior(f_K):
    lo = (2 * ELLS.min() + 1) / (2 * ELLS.min() - 3) * f_K[0]
    hi = (2 * ELLS.min() + 1) / (2 * ELLS.min() + 5) * f_K[-1]
    return (f_K > lo) & (f_K < hi)


def test_photo_rsd_flat_coefficients():
    S = np.ones((1, len(CHI)))
    ell = ELLS[:, None]
    A = (2 * ell**2 + 2 * ell - 1) / ((2 * ell - 1) * (2 * ell + 3))
    B = -ell * (ell - 1) / ((2 * ell - 1) * np.sqrt((2 * ell - 3) * (2 * ell + 1)))
    C = (
        -(ell + 1)
        * (ell + 2)
        / ((2 * ell + 3) * np.sqrt((2 * ell + 1) * (2 * ell + 5)))
    )
    interior = _rsd_interior(CHI)
    rsd = np.asarray(get_photo_rsd(ELLS, CHI, S))[:, 0, interior]
    assert_allclose(rsd, np.broadcast_to(A + B + C, rsd.shape), rtol=1e-12, atol=1e-14)
