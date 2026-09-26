"""Backwards-compatibility shim.

The single-bin and multi-bin modified-gravity perturbations have been merged into
:mod:`cloelib.cosmology.mg_cosmopower_jax_cosmology`. This module re-exports the
public API so existing imports keep working::

    from cloelib.cosmology.mg_multibin_cosmopower_jax_cosmology import (
        MGParams, multibin_mg_perturbations,
    )

New code should import from ``mg_cosmopower_jax_cosmology`` directly.
"""

from cloelib.cosmology.mg_cosmopower_jax_cosmology import (  # noqa: F401
    MGParams,
    mg_perturbations,
    binned_mg_perturbations,
    multibin_mg_perturbations,
    N_BINS,
)

__all__ = [
    "MGParams",
    "mg_perturbations",
    "binned_mg_perturbations",
    "multibin_mg_perturbations",
    "N_BINS",
]
