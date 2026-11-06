r"""Beyond-Limber angular power spectrum of two sets of contributions.

Assembles the building blocks of `cloelib.auxiliary.nonlimber_kernels` into
the *linear* beyond-Limber $C_\ell$ of two tracers, on the multipole nodes of
the T-tilde dataset:
$$
C_\ell^{XY} = \sum_{a \in X} \sum_{b \in Y} C_\ell^{ab},
$$
with $C_\ell^{ab}$ the double $(\chi, R)$ integral of `pair_integral` for the
contributions $a$ and $b$ (e.g. galaxy bias, magnification, RSD, lensing,
NLA). What each contribution brings is read from its `nonlimber_spec`
(`NonLimberSpec`); the cosmology-dependent part (the projected densities $w_\ell$) comes from a
`SpectraBank` built with `t_tildes`, shared by every pair.

`interpolate_correction` brings the beyond-Limber correction
$\Delta C_\ell = C_\ell^{\rm nonLimber} - C_\ell^{\rm Limber}$ (both linear)
from the nodes to the requested multipoles.
"""

from typing import Callable

import interpax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cloelib.auxiliary.nonlimber_kernels import kernel_on_chi_R, pair_integral
from cloelib.observables.photo.contributions import (
    contribution_field,
    nonlimber_spec_of,
    t_tilde_names,
)
from cloelib.observables.photo.spectrum_engine import SpectraBank


def nonlimber_cl(
    contributions1,
    contributions2,
    z: Array,
    bank: SpectraBank,
    raw_kernel: Callable | None = None,
) -> Array:
    r"""Linear beyond-Limber $C_\ell$ of two sets of contributions, on the T-tilde nodes.

    For every pair $(a, b)$ of contributions: the two T-tilde matrices of
    the pair (`t_tilde_names`), the projected densities of the two orderings
    ($a$ at $\chi$ and $b$ at $R\chi$, and vice versa, each with the fields
    the contributions trace), the kernels on the $(\chi, R\chi)$ grids and
    the product of the two $\ell$-dependent prefactors, then
    `pair_integral`. The kernels of a contribution are computed once, also
    when the same contribution object appears in both sets.

    The kernels are the raw `compute_kernel` ones: tracer-level factors
    such as the multiplicative shear bias $1 + m$ are *not* applied.

    Args:
      contributions1: contributions of the first tracer (e.g.
        `nonlimber_contributions(tracer1)`).
      contributions2: contributions of the second tracer.
      z (Array): redshift grid the kernels are computed on, shape `(n_z,)`.
        The kernels vanish outside it.
      bank (SpectraBank): bank built with `t_tildes`; it provides the grids,
        $z(\chi)$ and the projected densities $w_\ell$.
      raw_kernel (callable, optional): `contribution -> kernel on z`, e.g.
        a cache shared with the Limber windows; defaults to
        `contribution.compute_kernel(z)`.

    Returns:
      Array: $C_\ell$ at `bank.t_tildes.grid.ells`, shape
        `(n_ell, n_bins1, n_bins2)`.

    Raises:
      NotImplementedError: if a contribution has no `nonlimber_spec`.
    """
    grid = bank.t_tildes.grid
    z_of_chi = bank.z_of_chi()
    kernels = {}
    if raw_kernel is None:

        def raw_kernel(contribution):
            return contribution.compute_kernel(z)

    def kernels_of(contribution, spec):
        key = id(contribution)
        if key not in kernels:
            kernels[key] = kernel_on_chi_R(
                raw_kernel(contribution),
                z,
                z_of_chi,
                grid.chi,
                grid.R,
                spec.divide_by_chi2,
            )
        return kernels[key]

    C_ell = 0.0
    for c1 in contributions1:
        spec1 = nonlimber_spec_of(c1)
        field1 = contribution_field(c1)
        K1, K1_R = kernels_of(c1, spec1)
        for c2 in contributions2:
            spec2 = nonlimber_spec_of(c2)
            field2 = contribution_field(c2)
            K2, K2_R = kernels_of(c2, spec2)
            name_12, name_21 = t_tilde_names(spec1, spec2)
            C_ell = C_ell + pair_integral(
                K1,
                K2,
                K1_R,
                K2_R,
                bank.projected_density(name_12, field1, field2),
                bank.projected_density(name_21, field2, field1),
                spec1.ell_prefactor(grid.ells) * spec2.ell_prefactor(grid.ells),
                grid.w_chi,
                grid.w_R,
            )
    return C_ell


def interpolate_correction(
    delta: Array, ell_nodes: Array, ells: Array, ell_max: float
) -> Array:
    r"""Beyond-Limber correction $\Delta C_\ell$ from the nodes to `ells`.

    Akima interpolation in $\log \ell$ of the values (not of their log:
    cross spectra can change sign). The correction is applied on
    `[ell_nodes[0], ell_max]` and is zero outside it: above `ell_max` the
    Limber approximation is kept, and below the first node there is no
    beyond-Limber information (the standard dataset starts at $\ell = 2$,
    so $\ell = 0, 1$ keep the Limber value). A dataset with more nodes,
    e.g. at low integer multipoles, extends the range with no other change.

    The queries are clipped to that range before the interpolation, so that
    the masked ones (e.g. $\ell = 0$, where $\log \ell$ diverges) are
    finite and do not spoil `jax.grad` through `jnp.where`.

    Args:
      delta (Array): $\Delta C_\ell$ on the nodes, shape `(n_nodes, ...)`.
      ell_nodes (Array): increasing multipole nodes, shape `(n_nodes,)`.
        Must be concrete (not traced): it fixes the validity range.
      ells (Array): multipoles to evaluate the correction at, shape `(n_ell,)`.
      ell_max (float): largest multipole the correction is applied at.

    Returns:
      Array: shape `(n_ell, ...)`.

    Raises:
      ValueError: if `ell_max` is outside `[ell_nodes[0], ell_nodes[-1]]`.
    """
    ell_nodes = np.asarray(ell_nodes)
    ell_min, ell_last = float(ell_nodes[0]), float(ell_nodes[-1])
    if not ell_min <= ell_max <= ell_last:
        raise ValueError(
            f"ell_max = {ell_max} must lie within the beyond-Limber nodes "
            f"[{ell_min}, {ell_last}]."
        )
    ells = jnp.asarray(ells)
    inside = (ells >= ell_min) & (ells <= ell_max)
    log_ells = jnp.log(jnp.clip(ells, ell_min, ell_max))
    values = interpax.interp1d(log_ells, np.log(ell_nodes), delta, method="akima")
    mask = inside.reshape(inside.shape + (1,) * (values.ndim - 1))
    return jnp.where(mask, values, 0.0)
