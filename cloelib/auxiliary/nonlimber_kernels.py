r"""Pure array-level building blocks of the beyond-Limber angular power spectrum.

The beyond-Limber C_ell of a pair of kernels (A, B) is written as

$$
C_\ell^{AB} = \frac{2}{\pi}\,\mathcal{P}_\ell \int \mathrm{d}\chi\,\chi
\int_0^1 \mathrm{d}R\;
\Big[ K_A(\chi)\,K_B(R\chi)\,w^{12}_\ell(\chi, R)
     + K_B(\chi)\,K_A(R\chi)\,w^{21}_\ell(\chi, R) \Big],
$$

where $w_\ell(\chi, R)$ is obtained by contracting the Chebyshev expansion
(in $\log k$) of the unequal-time power spectrum $P(k, \chi, R\chi)$ with
the precomputed T-tilde matrices, and $\mathcal{P}_\ell$ collects the
$\ell$-dependent prefactors of the two kernels.

Pipeline, one function per step:

1. `unequal_time_pk`: $P(k,\chi,R\chi)$ from the equal-time $P(k,\chi)$.
2. `chebyshev.Pkl_chebyshev_coeffs`: Chebyshev coefficients in $\log k$.
3. `w_ell`: contraction with a T-tilde matrix.
4. `kernel_on_chi_R`: kernels on the $(\chi, R\chi)$ grids.
5. `pair_integral`: the double integral above for one pair of kernels.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cloelib.auxiliary.akima import akima_interpolation

jax.config.update("jax_enable_x64", True)


@jax.jit
def unequal_time_pk(Pk_chi: Array, chi: Array, R: Array) -> Array:
    r"""Unequal-time matter power spectrum, as the geometric mean of two equal-time ones.

    $$
    P(k, \chi, R\chi) = \sqrt{P(k, \chi)\,P(k, R\chi)}
    $$

    $P(k, R\chi)$ is obtained by Akima interpolation of $\log_{10} P$ along
    `chi` (extrapolating where $R\chi$ falls below `chi[0]`).

    Parameters:
      Pk_chi (Array): Equal-time (strictly positive) power spectrum, shape `(len(chi), n_k)`.
      chi (Array): Comoving distance grid, shape `(n_chi,)`.
      R (Array): Ratios $\chi_2/\chi_1$, shape `(n_R,)`.

    Returns:
      Array: $P(k, \chi, R\chi)$, shape `(n_k, n_chi, n_R)`.
    """
    chi_R = jnp.outer(chi, R).flatten()
    Pk_chi_R = 10 ** akima_interpolation(jnp.log10(Pk_chi), chi, chi_R, axis=0)
    Pk_chi_R = Pk_chi_R.reshape(len(chi), len(R), Pk_chi.shape[1])
    return jnp.sqrt(jnp.einsum("ck,crk->kcr", Pk_chi, Pk_chi_R))


@jax.jit
def w_ell(cheb_coeffs: Array, T_tilde: Array) -> Array:
    r"""Contract the Chebyshev coefficients of the Pk with a T-tilde matrix.

    Parameters:
      cheb_coeffs (Array): Chebyshev coefficients of $P(k,\chi,R\chi)$ in
        $\log k$, shape `(n_cheb, n_chi, n_R)`
        (output of `chebyshev.Pkl_chebyshev_coeffs`).
      T_tilde (Array): Shape `(n_ell, n_chi, n_R, n_cheb)`.

    Returns:
      Array: $w_\ell(\chi, R)$, shape `(n_ell, n_chi, n_R)`.
    """
    return jnp.einsum("ijk,ljki->ljk", cheb_coeffs, T_tilde)


@partial(jax.jit, static_argnames=("divide_by_chi2",))
def kernel_on_chi_R(
    W_z: Array,
    z: Array,
    z_of_chi: Array,
    chi: Array,
    R: Array,
    divide_by_chi2: bool,
) -> tuple[Array, Array]:
    r"""Resample a kernel from the redshift grid to the $\chi$ and $R\chi$ grids.

    The kernel is first interpolated to `z_of_chi` (and set to zero outside
    `[z[0], z[-1]]`), optionally divided by $\chi^2$ (kernels defined per unit
    $\chi^2$, e.g. lensing; positions are not), then interpolated to $R\chi$.

    Parameters:
      W_z (Array): Kernel on the redshift grid, shape `(n_bins, len(z))`.
      z (Array): Redshift grid `W_z` is defined on, shape `(len(z),)`.
      z_of_chi (Array): Redshift corresponding to each `chi`, shape `(n_chi,)`.
      chi (Array): Comoving distance grid, shape `(n_chi,)`.
      R (Array): Ratios $\chi_2/\chi_1$, shape `(n_R,)`.
      divide_by_chi2 (bool): Static. Whether to divide the kernel by $\chi^2$.

    Returns:
      tuple[Array, Array]: $K(\chi)$ with shape `(n_bins, n_chi)` and
        $K(R\chi)$ with shape `(n_bins, n_chi, n_R)`.
    """
    W_chi = akima_interpolation(W_z, z, z_of_chi, axis=-1)
    # Beyond the redshift range the kernel was built on there is no signal:
    # an Akima extrapolation there would be arbitrary, not physical.
    inside = (z_of_chi >= z[0]) & (z_of_chi <= z[-1])
    W_chi = jnp.where(inside, W_chi, 0.0)
    if divide_by_chi2:
        W_chi = W_chi / chi**2

    chi_R = jnp.outer(chi, R).flatten()
    W_chi_R = akima_interpolation(W_chi, chi, chi_R, axis=-1)
    return W_chi, W_chi_R.reshape(W_chi.shape[0], len(chi), len(R))


@jax.jit
def pair_integral(
    K_A: Array,
    K_B: Array,
    K_A_R: Array,
    K_B_R: Array,
    w_12: Array,
    w_21: Array,
    ell_prefactor: Array,
    w_chi: Array,
    w_R: Array,
) -> Array:
    r"""Beyond-Limber $C_\ell$ of one pair of kernels (A, B).

    The $\chi$ and $R$ integrals and the sum over the two orderings are
    fused into a single contraction, so the five-index
    `(n_ell, n_bin_A, n_bin_B, n_chi, n_R)` integrand is never materialized.

    Parameters:
      K_A, K_B (Array): $K(\chi)$ of each kernel, shape `(n_bin, n_chi)`.
      K_A_R, K_B_R (Array): $K(R\chi)$ of each kernel, shape `(n_bin, n_chi, n_R)`.
      w_12, w_21 (Array): $w_\ell(\chi, R)$ of the two orderings, shape
        `(n_ell, n_chi, n_R)`. Equal for symmetric pairs; they differ only
        for pairs involving RSD.
      ell_prefactor (Array): Product of the two kernels' $\ell$-dependent
        prefactors, shape `(n_ell,)`.
      w_chi (Array): Quadrature weights of the $\chi$ integral, *including*
        the $\chi$ measure factor (`chi * simpson * dchi`), shape `(n_chi,)`.
      w_R (Array): Quadrature weights of the $R$ integral, shape `(n_R,)`.

    Returns:
      Array: $C_\ell$, shape `(n_ell, n_bin_A, n_bin_B)`.
    """
    left = jnp.einsum(
        "l,ik,jkr,lkr,k,r->lij",
        ell_prefactor,
        K_A,
        K_B_R,
        w_12,
        w_chi,
        w_R,
        optimize="optimal",
    )
    right = jnp.einsum(
        "l,jk,ikr,lkr,k,r->lij",
        ell_prefactor,
        K_B,
        K_A_R,
        w_21,
        w_chi,
        w_R,
        optimize="optimal",
    )
    return 2.0 / jnp.pi * (left + right)
