import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.auxiliary.chebyshev import (
    chebyshev_points_interval,
    Pkl_chebyshev_coeffs,
)
from cloelib.auxiliary.math_utils import simpsons_weights_jit
from cloelib.auxiliary.nonlimber_kernels import (
    kernel_on_chi_R,
    pair_integral,
    unequal_time_pk,
    w_ell,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# unequal_time_pk
# ---------------------------------------------------------------------------
def test_unequal_time_pk_matches_geometric_mean():
    """The unequal-time spectrum is the geometric mean of two equal-time ones.

    P(k, chi, R chi) must equal sqrt(P(k, chi) P(k, R chi)). Taking log10 P
    linear in chi makes the Akima interpolation of P(k, R chi) exact, even
    below chi[0] where it extrapolates, so the closed form
    k^-1 10^(-chi (1 + R) / 1000) is matched to rounding error.
    """
    ks = jnp.logspace(-3, 0, 7)
    chi = jnp.linspace(200.0, 2000.0, 40)
    R = jnp.linspace(0.1, 1.0, 9)
    Pk_chi = (1.0 / ks)[None, :] * 10 ** (-chi[:, None] / 500.0)

    out = unequal_time_pk(Pk_chi, chi, R)

    expected = (1.0 / ks)[:, None, None] * 10 ** (
        -(chi[None, :, None] * (1.0 + R[None, None, :])) / 1000.0
    )
    assert out.shape == (len(ks), len(chi), len(R))
    np.testing.assert_allclose(out, expected, rtol=1e-8)


def test_unequal_time_pk_equal_time_limit_and_symmetry():
    """At R = 1 the two times coincide, so the geometric mean is P(k, chi) itself.

    Also checks the output layout: (k, chi, R), i.e. the transpose of the
    (chi, k) input.
    """
    ks = jnp.logspace(-3, 0, 5)
    chi = jnp.linspace(100.0, 1000.0, 50)
    Pk_chi = (1.0 / ks)[None, :] * (1.0 + 1e-3 * chi[:, None])
    out = unequal_time_pk(Pk_chi, chi, jnp.array([1.0]))
    # R = 1 is the equal-time spectrum
    np.testing.assert_allclose(out[:, :, 0], Pk_chi.T, rtol=1e-10)


# ---------------------------------------------------------------------------
# w_ell
# ---------------------------------------------------------------------------
def test_w_ell_matches_explicit_loop():
    """w_ell(chi, R) is the sum over Chebyshev index n of c_n(chi, R) T_tilde(ell, chi, R, n).

    Compared with explicit Python loops on random data: guards the einsum
    index order, which silently transposes if axes are swapped.
    """
    rng = np.random.default_rng(0)
    n_cheb, n_chi, n_R, n_ell = 6, 5, 4, 3
    c = rng.normal(size=(n_cheb, n_chi, n_R))
    T = rng.normal(size=(n_ell, n_chi, n_R, n_cheb))

    expected = np.zeros((n_ell, n_chi, n_R))
    for ell in range(n_ell):
        for j in range(n_chi):
            for k in range(n_R):
                expected[ell, j, k] = np.sum(c[:, j, k] * T[ell, j, k, :])

    out = w_ell(jnp.asarray(c), jnp.asarray(T))
    assert out.shape == (n_ell, n_chi, n_R)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_chebyshev_coeffs_pipeline_shapes():
    """unequal_time_pk and Pkl_chebyshev_coeffs fit together.

    The output of the first (n_k, n_chi, n_R) is the input of the second;
    with 32 + 1 nodes it must return (33, n_chi, n_R) coefficients, the layout
    w_ell and the T-tilde matrices (n_cheb last) expect.
    """
    ks = jnp.logspace(-4, 1, 300)
    chi = jnp.linspace(100.0, 1000.0, 12)
    R = jnp.linspace(0.1, 1.0, 5)
    Pk_chi = (1.0 / ks)[None, :] * jnp.ones((len(chi), 1))
    k_cheb = chebyshev_points_interval(32, -3.0, 0.0)  # log10 k nodes

    Pk = unequal_time_pk(Pk_chi, chi, R)
    coeffs = Pkl_chebyshev_coeffs(Pk, jnp.log10(ks), k_cheb)
    assert coeffs.shape == (33, len(chi), len(R))


# ---------------------------------------------------------------------------
# kernel_on_chi_R
# ---------------------------------------------------------------------------
@pytest.fixture
def linear_kernel_setup():
    """Kernels linear in z, W(z) = a z + b, with z(chi) = chi / 250.

    Akima interpolation is exact on linear data, even when extrapolating, so
    the expected values of the tests below are known in closed form.
    """
    z = jnp.linspace(0.0, 2.0, 60)
    chi = jnp.linspace(100.0, 500.0, 50)
    z_of_chi = chi / 250.0  # chi in [100, 500] -> z in [0.4, 2.0]
    a = jnp.array([1.0, -2.0, 0.5])
    b = jnp.array([0.3, 1.0, 2.0])
    W_z = a[:, None] * z[None, :] + b[:, None]
    return z, chi, z_of_chi, a, b, W_z


def test_kernel_on_chi_R_linear_exact(linear_kernel_setup):
    """A kernel is correctly resampled from the z grid to the chi and R chi grids.

    Without the chi^-2 factor, a kernel linear in z is linear in chi, so both
    K(chi) and K(R chi) have a closed form (the latter also below chi[0], by
    exact extrapolation). Checks shapes and the (bin, chi, R) axis order.
    """
    z, chi, z_of_chi, a, b, W_z = linear_kernel_setup
    R = jnp.linspace(0.3, 1.0, 6)

    W_chi, W_chi_R = kernel_on_chi_R(W_z, z, z_of_chi, chi, R, divide_by_chi2=False)

    exp_chi = a[:, None] * z_of_chi[None, :] + b[:, None]
    exp_chi_R = (
        a[:, None, None] * (chi[None, :, None] * R[None, None, :] / 250.0)
        + b[:, None, None]
    )
    assert W_chi.shape == (3, len(chi))
    assert W_chi_R.shape == (3, len(chi), len(R))
    np.testing.assert_allclose(W_chi, exp_chi, atol=1e-10)
    np.testing.assert_allclose(W_chi_R, exp_chi_R, atol=1e-9)


def test_kernel_on_chi_R_divide_by_chi2(linear_kernel_setup):
    """divide_by_chi2 turns K(z(chi)) into K(z(chi)) / chi^2 (lensing-like kernels).

    At R = 1, R chi is a node of the chi grid, so K(R chi) must be exactly
    K(chi): this ties the second interpolation to the first without relying on
    interpolation accuracy.
    """
    z, chi, z_of_chi, a, b, W_z = linear_kernel_setup
    R = jnp.array([0.9, 1.0])

    W_plain, _ = kernel_on_chi_R(W_z, z, z_of_chi, chi, R, divide_by_chi2=False)
    W_div, W_div_R = kernel_on_chi_R(W_z, z, z_of_chi, chi, R, divide_by_chi2=True)

    np.testing.assert_allclose(W_div, W_plain / chi**2, rtol=1e-12)
    # R = 1 is a node of the chi grid: exact, no interpolation error
    np.testing.assert_allclose(W_div_R[:, :, -1], W_div, rtol=1e-10)


def test_kernel_on_chi_R_zero_outside_redshift_range(linear_kernel_setup):
    """There is no signal beyond the redshift range the kernel was built on.

    Where z(chi) exceeds z[-1], an Akima extrapolation would give arbitrary
    values; the kernel is set to zero there instead, and left untouched inside.
    """
    z, _, _, a, b, W_z = linear_kernel_setup
    chi = jnp.linspace(100.0, 600.0, 50)
    z_of_chi = chi / 250.0  # chi > 500 maps beyond z[-1] = 2
    R = jnp.array([1.0])

    W_chi, _ = kernel_on_chi_R(W_z, z, z_of_chi, chi, R, divide_by_chi2=False)

    outside = np.asarray(z_of_chi > z[-1])
    assert outside.any() and (~outside).any()
    np.testing.assert_array_equal(np.asarray(W_chi)[:, outside], 0.0)
    assert np.all(np.abs(np.asarray(W_chi)[:, ~outside]) > 0.0)


# ---------------------------------------------------------------------------
# pair_integral
# ---------------------------------------------------------------------------
def _random_pair_inputs(n_ell=4, n_a=3, n_b=2, n_chi=7, n_R=5, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "K_A": jnp.asarray(rng.normal(size=(n_a, n_chi))),
        "K_B": jnp.asarray(rng.normal(size=(n_b, n_chi))),
        "K_A_R": jnp.asarray(rng.normal(size=(n_a, n_chi, n_R))),
        "K_B_R": jnp.asarray(rng.normal(size=(n_b, n_chi, n_R))),
        "w_12": jnp.asarray(rng.normal(size=(n_ell, n_chi, n_R))),
        "w_21": jnp.asarray(rng.normal(size=(n_ell, n_chi, n_R))),
        "ell_prefactor": jnp.asarray(rng.uniform(0.5, 2.0, size=n_ell)),
        "w_chi": jnp.asarray(rng.uniform(0.5, 2.0, size=n_chi)),
        "w_R": jnp.asarray(rng.uniform(0.5, 2.0, size=n_R)),
    }


def _pair_integral_reference(
    K_A, K_B, K_A_R, K_B_R, w_12, w_21, ell_prefactor, w_chi, w_R
):
    """Straightforward five-index implementation of the same formula."""
    left = jnp.einsum("ik,jkr,lkr->lijkr", K_A, K_B_R, w_12)
    right = jnp.einsum("jk,ikr,lkr->lijkr", K_B, K_A_R, w_21)
    integrand = jnp.einsum("l,lijkr->lijkr", ell_prefactor, left + right)
    return 2.0 / jnp.pi * jnp.einsum("lijkr,k,r->lij", integrand, w_chi, w_R)


def test_pair_integral_matches_reference():
    """The fused contraction equals the textbook five-index formula.

    The reference builds the full (ell, i, j, chi, R) integrand and sums it
    afterwards; pair_integral must give the same numbers from random inputs,
    including different weights for the two orderings (w_12 != w_21, as for RSD).
    """
    inputs = _random_pair_inputs()
    out = pair_integral(**inputs)
    assert out.shape == (4, 3, 2)
    np.testing.assert_allclose(
        out, _pair_integral_reference(**inputs), rtol=1e-10, atol=1e-12
    )


def test_pair_integral_symmetric_pair_is_symmetric_in_bins():
    """With A = B (same kernels, w_12 = w_21) C_ell[i, j] = C_ell[j, i].

    A property of the physics, not of the implementation: it fails if the two
    orderings are not combined consistently.
    """
    inputs = _random_pair_inputs(n_a=4, n_b=4)
    inputs["K_B"], inputs["K_B_R"] = inputs["K_A"], inputs["K_A_R"]
    inputs["w_21"] = inputs["w_12"]

    out = np.asarray(pair_integral(**inputs))
    np.testing.assert_allclose(out, np.swapaxes(out, 1, 2), rtol=1e-10)


def test_pair_integral_analytic_constant_integrand():
    """Constant integrand: C_ell = 2/pi * 2 * prefactor_ell * (int dchi) * (int dR).

    With unit kernels and w_ell, chi in [0, 4] and R in [0, 1] (Simpson
    weights), this pins the overall normalization, including the 2/pi and the
    factor 2 from adding the two orderings.
    """
    n_ell, n_bin, n_chi, n_R = 3, 2, 21, 11
    chi = jnp.linspace(0.0, 4.0, n_chi)
    R = jnp.linspace(0.0, 1.0, n_R)
    w_chi = simpsons_weights_jit(n_chi) * (chi[1] - chi[0])
    w_R = simpsons_weights_jit(n_R) * (R[1] - R[0])
    pref = jnp.array([1.0, 2.0, 3.0])

    out = pair_integral(
        K_A=jnp.ones((n_bin, n_chi)),
        K_B=jnp.ones((n_bin, n_chi)),
        K_A_R=jnp.ones((n_bin, n_chi, n_R)),
        K_B_R=jnp.ones((n_bin, n_chi, n_R)),
        w_12=jnp.ones((n_ell, n_chi, n_R)),
        w_21=jnp.ones((n_ell, n_chi, n_R)),
        ell_prefactor=pref,
        w_chi=w_chi,
        w_R=w_R,
    )

    # 2/pi * (1 + 1) * pref_l * int_0^4 dchi * int_0^1 dR
    expected = 2.0 / jnp.pi * 2.0 * pref * 4.0 * 1.0
    np.testing.assert_allclose(
        out, expected[:, None, None] * np.ones((1, 2, 2)), rtol=1e-10
    )


def _jaxpr_max_intermediate_size(jaxpr):
    sizes = [0]
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(var.aval, "shape", ())
            sizes.append(int(np.prod(shape)) if shape else 1)
        for sub in jax.core.jaxprs_in_params(eqn.params):
            sizes.append(_jaxpr_max_intermediate_size(sub))
    return max(sizes)


def test_pair_integral_never_builds_the_five_index_tensor():
    """Memory: the (n_ell, n_i, n_j, n_chi, n_R) integrand is never materialized.

    It would dominate memory for a realistic 3x2 (about 180 MB per pair). The
    jaxpr is inspected: no intermediate array may reach that size. The same
    check fails for the textbook implementation, so it does detect the problem.
    """
    n_ell, n_a, n_b, n_chi, n_R = 4, 5, 6, 7, 8
    inputs = _random_pair_inputs(n_ell, n_a, n_b, n_chi, n_R)

    jaxpr = jax.make_jaxpr(pair_integral)(**inputs).jaxpr

    full_integrand = n_ell * n_a * n_b * n_chi * n_R
    assert _jaxpr_max_intermediate_size(jaxpr) < full_integrand


def test_pair_integral_is_jittable_and_differentiable():
    """pair_integral is usable inside a jitted, differentiated likelihood.

    jax.value_and_grad of a scalar function of w_12 must trace and give a
    finite value and gradient.
    """
    inputs = _random_pair_inputs()

    def total(scale):
        return pair_integral(**{**inputs, "w_12": inputs["w_12"] * scale}).sum()

    value, grad = jax.value_and_grad(total)(1.0)
    assert np.isfinite(value) and np.isfinite(grad)
