import numpy as onp
import pytest
from scipy.special import eval_jacobi, eval_legendre

from cloelib.summary_statistics import angular_correlation_function_wigner


def test_wigner_cache():
    beta = 1.0
    ell = 200

    assert angular_correlation_function_wigner.d_0_0_ell(
        beta, ell
    ) == angular_correlation_function_wigner._d_0_0_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_0_ell(
        beta, ell
    ) == angular_correlation_function_wigner._d_2_0_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_2_ell(
        beta, ell
    ) == angular_correlation_function_wigner._d_2_2_ell_compute(beta, ell)
    assert angular_correlation_function_wigner.d_2_m2_ell(
        beta, ell
    ) == angular_correlation_function_wigner._d_2_m2_ell_compute(beta, ell)


def _d_reference(element, beta, ell):
    """Wigner small-d elements from their Jacobi-polynomial representation."""
    c, s, x = onp.cos(beta / 2), onp.sin(beta / 2), onp.cos(beta)
    k = ell - 2
    if element == "d_0_0":
        return eval_legendre(ell, x)
    if element == "d_2_2":
        return c**4 * eval_jacobi(k, 0, 4, x)
    if element == "d_2_m2":
        return s**4 * eval_jacobi(k, 4, 0, x)
    if element == "d_2_0":
        norm = onp.sqrt((ell + 2) * (ell + 1) / (ell * (ell - 1)))
        return norm * s**2 * c**2 * eval_jacobi(k, 2, 2, x)
    raise ValueError(element)


@pytest.mark.parametrize("element", ["d_0_0", "d_2_2", "d_2_m2", "d_2_0"])
def test_wigner_vmap_against_jacobi(element):
    # arcmin-scale angles used for 2PCFs, and multipoles beyond 30000
    theta = onp.radians(onp.geomspace(0.5, 300.0, 7) / 60)
    ells = onp.arange(2, 40001)
    ells_check = onp.array([2, 3, 4, 10, 200, 2999, 29999, 30000, 35000, 40000])

    d_vmap = getattr(angular_correlation_function_wigner, f"{element}_vmap")
    d = onp.asarray(d_vmap(theta, ells))
    assert d.shape == (len(theta), len(ells))

    ref = _d_reference(element, theta[:, None], ells_check[None, :])
    onp.testing.assert_allclose(d[:, ells_check - 2], ref, rtol=0, atol=1e-10)

    # selecting a non-contiguous subset of multipoles gives the same values
    d_subset = onp.asarray(d_vmap(theta, ells_check))
    onp.testing.assert_allclose(d_subset, d[:, ells_check - 2], rtol=0, atol=0)


@pytest.mark.parametrize("element", ["d_0_0", "d_2_2", "d_2_m2", "d_2_0"])
def test_wigner_ell_matches_vmap(element):
    theta = onp.array([0.01, 0.5, 1.0])
    ells = onp.arange(2, 301)
    d_vmap = getattr(angular_correlation_function_wigner, f"{element}_vmap")(
        theta, ells
    )
    d_ell = getattr(angular_correlation_function_wigner, f"_{element}_ell_compute")
    for ell in [2, 3, 4, 57, 300]:
        onp.testing.assert_allclose(
            onp.asarray(d_ell(theta, ell)), d_vmap[:, ell - 2], rtol=1e-12, atol=1e-14
        )
