"""
Implementation of the GODMAX halo model on top of cloelib's JAX cosmology.

## Design

**``GODMAXNonLinearPerturbations``** runs the GODMAX halo model on cloelib's
JAX background, Eisenstein-Hu linear P(k) and halofit, and provides:

- the non-linear matter power spectrum: cloelib's halofit times the GODMAX
  baryonic response P_DMB / P_NFW;
- the halo-model spectra of matter (``"m"``), Compton-y (``"y"``, tSZ) and HOD
  galaxies (``"g"``), through ``power_spectrum(field1, field2, zs, ks)``. These
  are used by `cloelib.observables.cmb.tsz.tSZTracer` and by
  `PositionsTracer(..., galaxy_bias_model="hod")`.

## Units

GODMAX works in h-units (k in h/Mpc, P in (Mpc/h)^3, distances in Mpc/h);
cloelib works in Mpc (k in 1/Mpc, P in Mpc^3). The conversion is done in this
module only (`_spectra_in_cloelib_units`).

## Per-bin HOD

Any HOD parameter in `galaxies["hod"]` can be an array of length n_bins + 1
instead of a scalar. The first entry is the default; entry i + 1 applies to the
true-redshift interval `galaxies["z_edges"][i]`. Scalars and arrays can be mixed.

This is one galaxy field whose HOD changes piecewise with true redshift, not
independent galaxy populations per tomographic bin: every tomographic n(z), and
every cross-bin pair, uses the same P_gg(k, z) at a given z. It equals an HOD per
tomographic bin only when the bins do not overlap in true redshift.

The intervals are passed to GODMAX unchanged, and GODMAX assigns them as follows:

- edges are exclusive (lower < z < upper);
- redshifts below the first interval, above the last, in a gap between intervals
  or exactly on an edge use the default entry;
- where intervals overlap, the later interval's values are used.

The interval is chosen at each halo-model redshift node (`redshifts`); between
nodes the spectra are interpolated, so a change of HOD at an edge is spread over
one node spacing. Place nodes close to the edges if the transition matters.

## Halo-model spectra

Each spectrum is the sum of a 1-halo and a 2-halo term, combined in one of two
ways, chosen per spectrum in `godmax_config["analysis"]`:

- `poweradd` (default): P = (P_1h^alpha + P_2h^alpha)^(1/alpha); alpha = 1 (the
  default) is the plain sum P_1h + P_2h;
- `response`: the `poweradd` result times P_halofit / P_NFW, where P_NFW is the
  gravity-only (NFW) halo-model matter spectrum.

| Spectrum | Fields | Prescription | Parameters |
|---|---|---|---|
| matter-matter | `"m", "m"` | cloelib halofit x P_DMB / P_NFW, each a plain 1h + 2h sum | halo and gas parameters (`SIM_PARAMETERS`) |
| matter-y | `"m", "y"` | `poweradd` or `response` | `alpha_ky`, `tSZ_transition_model` |
| y-y | `"y", "y"` | always the plain sum P_1h + P_2h | none |
| galaxy-matter | `"g", "m"` | `poweradd` or `response` | `alpha_gm`, `gm_transition_model` |
| galaxy-y | `"g", "y"` | `poweradd` or `response` | `alpha_gy`, `tSZ_transition_model` |
| galaxy-galaxy | `"g", "g"` | `poweradd`; `response` is the plain sum x P_halofit / P_NFW | `alpha_gg` (unused with `response`), `gg_transition_model` |

Notes:

- `alpha_ky` and `tSZ_transition_model` do not affect y-y.
- `tSZ_transition_model` sets both matter-y and galaxy-y.
- The matter leg of matter-y is baryonified (DMB) with
  `baryonification_tSZ=True` (default) and gravity-only (NFW) otherwise.
- `alpha_ge` is passed to GODMAX but has no effect here: cloelib does not use
  the galaxy-electron spectrum.
- The alpha values are fixed settings: GODMAX converts them with `float()`, so
  they cannot be JAX-traced (sampled with gradients or varied inside `jit`).

## Notes

- GODMAX is not on PyPI. Install it from GitHub (see the README); it needs the
  ``godmax.helpers.cloelib_cosmology`` module.
- GODMAX is flat w0CDM without massive neutrinos.
- `DEFAULT_HALO_PARAMS` and `DEFAULT_ANALYSIS` are cloelib's own settings,
  not a copy of GODMAX's ``params_default.yaml`` (the mass range and
  `kthresh_lowpass_Pmm1h_lowk` differ from it).
- Halo-model, HOD and `alpha_*` parameters that are not given take GODMAX's
  code defaults (``base_class.py``), which might differ from
  ``params_default.yaml``. No fiducial values are fixed yet; pass the
  parameters explicitly to control the model.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cloelib.cosmology.cosmology import CB_CROSS_CORRELATED, BaryonBoostMixin
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)

#: Halo-model grids and models (cloelib's settings; see the module notes).
#: `z_array` is always set from the redshifts passed to the perturbations class.
DEFAULT_HALO_PARAMS: Dict[str, Any] = {
    "rmin": 0.005,
    "rmax": 8.0,
    "nr": 23,
    "lg10_Mmin": 10.5,
    "lg10_Mmax": 16.5,
    "nM": 24,
    "kmin": 1e-4,
    "kmax": 150.0,
    "nk": 48,
    "conc_model": "Duffy08",
    "hmf_model": "T10",
    "do_corr_2h_mm": True,
}

#: Analysis switches and accuracy settings (cloelib's settings; see the module
#: notes). `model_tSZ`, `model_galaxies` and the cosmology are
#: always set by the perturbations class.
DEFAULT_ANALYSIS: Dict[str, Any] = {
    "backreaction": True,
    "baryonification_tSZ": True,
    "num_points_trapz_int": 32,
    "num_points_gal_cal": 32,
    "lowpass_Pmm1h_lowk": True,
    "kthresh_lowpass_Pmm1h_lowk": 1e-3,
    "tSZ_transition_model": "poweradd",
    "gg_transition_model": "poweradd",
}

#: GODMAX `sim_params` keys accepted as keyword arguments (gas, backreaction,
#: non-thermal pressure and simple stellar profile parameters).
SIM_PARAMETERS = (
    "nfw_trunc",
    "epsilon_rt",
    "theta_ej_0",
    "log10_Mstar0_theta_ej",
    "nu_theta_ej_M",
    "nu_theta_ej_z",
    "nu_theta_ej_c",
    "theta_co_0",
    "log10_Mstar0_theta_co",
    "nu_theta_co_M",
    "nu_theta_co_z",
    "nu_theta_co_c",
    "mu_beta",
    "log10_Mstar0",
    "log10_Mc0",
    "nu_z",
    "nu_M",
    "gamma_rhogas",
    "delta_rhogas",
    "a_zeta",
    "n_zeta",
    "alpha_nt",
    "beta_nt",
    "n_nt",
    "eta_star",
    "eta_cga",
    "A_starcga",
    "log10_M1_starcga",
)

#: GODMAX `other_params` keys accepted as keyword arguments (1-halo/2-halo
#: transition smoothing of each spectrum). See "Halo-model spectra" in the
#: module docstring for which spectrum each one controls.
OTHER_PARAMETERS = ("alpha_ky", "alpha_gy", "alpha_gg", "alpha_gm", "alpha_ge")

#: HOD parameters accepted in `galaxies["hod"]`; each is a scalar or, for a
#: per-bin HOD, an array of length n_bins + 1 (default value first). See
#: "Per-bin HOD" in the module docstring for how the bins are assigned.
#: Units are GODMAX's: `galaxies["nbar"]` is a comoving number density in
#: (h/Mpc)^3, unlike the Mpc units of the rest of cloelib. The SHMR masses
#: `log10M1_fshmr` and `log10Mstar0_fshmr` are log10 of masses in M_sun (no h);
#: `siglogMstar_Ncen` is in dex (a scatter in log10 M_star); `Bsat_Nsat` and
#: `Bcut_Nsat` are relative to a 1e12 M_sun pivot; the slopes, the `*_a_*`
#: redshift terms (per unit a - 1) and `fcen` are dimensionless. With
#: `godmax_config["analysis"]["Zheng_HOD"] = True`, the threshold is set by
#: `log10Mmin_Ncen` (log10 of a halo mass in M_sun/h) instead of `nbar`.
HOD_PARAMETERS = (
    "log10M1_fshmr",
    "log10M1_a_fshmr",
    "log10Mstar0_fshmr",
    "log10Mstar0_a_fshmr",
    "beta_fshmr",
    "beta_a_fshmr",
    "delta_fshmr",
    "delta_a_fshmr",
    "gamma_fshmr",
    "gamma_a_fshmr",
    "siglogMstar_Ncen",
    "alphasat_Nsat",
    "Bcut_Nsat",
    "Bsat_Nsat",
    "betacut_Nsat",
    "betasat_Nsat",
    "fcen",
    "log10Mmin_Ncen",
)

_GALAXIES_KEYS = ("nbar", "nbar_z", "z_edges", "hod")
_CONFIG_SECTIONS = ("sim_params", "halo_params", "analysis", "other_params")
#: `godmax_config` keys that cloelib sets itself, by section, and the argument
#: that sets them; giving them in `godmax_config` raises instead of being ignored.
_OWNED_KEYS = {
    "sim_params": {"cosmo": "background"},
    "halo_params": {"z_array": "redshifts"},
    "analysis": {
        "cosmology_backend": "background",
        "cosmology_obj": "background",
        "model_tSZ": "model_tSZ",
        "model_galaxies": "galaxies",
        "nbar_gal_comoving_val": "galaxies['nbar']",
        "nbar_gal_comoving_zarray": "galaxies['nbar_z']",
        "hod_params_model": "galaxies['hod']",
    },
}


def godmax_config_from_params(
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Use a GODMAX parameter set (e.g. a loaded GODMAX yaml) as `godmax_config`.

    Keys that cloelib sets itself (`_OWNED_KEYS`: cosmology, redshifts, backend,
    tSZ and galaxy switches, nbar, HOD mode) are dropped; pass those as arguments
    of `GODMAXNonLinearPerturbations`. The cosmology always comes from the cloelib
    background. Sections other than `_CONFIG_SECTIONS` are ignored.

    Args:
        params: GODMAX parameters by section (`sim_params`, `halo_params`,
            `analysis`, `other_params`).

    Returns:
        The parameters without the cloelib-owned keys, to pass as `godmax_config`.
    """
    return {
        section: {
            key: value
            for key, value in params.get(section, {}).items()
            if key not in _OWNED_KEYS.get(section, {})
        }
        for section in _CONFIG_SECTIONS
    }


#: (field1, field2) -> (GODMAX attribute, number of Compton-y legs).
_SPECTRA = {
    ("m", "m"): ("Pmm_tot_mat", 0),
    ("m", "y"): ("Pym_tot_mat", 1),
    ("y", "y"): ("Pyy_tot_kz_mat", 2),
    ("g", "g"): ("Pgg_tot_mat", 0),
    ("g", "m"): ("Pgm_tot_mat", 0),
    ("g", "y"): ("Pgy_tot_mat", 1),
}


def _galaxy_config(
    galaxies: Dict[str, Any], zheng_hod: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate the `galaxies` dict into GODMAX `sim_params` and `analysis` entries.

    Args:
        galaxies: Galaxy sample and HOD settings.
        zheng_hod: Whether GODMAX uses the Zheng HOD, whose threshold is set by
            `log10Mmin_Ncen` instead of `nbar`.
    """
    unknown = set(galaxies) - set(_GALAXIES_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown galaxies keys {sorted(unknown)}; expected {_GALAXIES_KEYS}."
        )
    if zheng_hod and ("nbar" in galaxies or "nbar_z" in galaxies):
        raise ValueError(
            "galaxies['nbar'] is not used with Zheng_HOD: the threshold is set by "
            "galaxies['hod']['log10Mmin_Ncen']."
        )
    if not zheng_hod and "nbar" not in galaxies:
        raise ValueError(
            "galaxies['nbar'] (comoving number density in (h/Mpc)^3) is required: "
            "GODMAX sets the stellar-mass threshold of the HOD from it."
        )

    sim: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {"model_galaxies": True}

    # nbar is given exactly when the HOD is not Zheng's (checked above).
    if "nbar" in galaxies:
        nbar = jnp.asarray(galaxies["nbar"], dtype=float)
        if nbar.ndim == 0:
            analysis["nbar_gal_comoving_val"] = float(nbar)
        else:
            if galaxies.get("nbar_z") is None:
                raise ValueError(
                    "galaxies['nbar_z'] is required when nbar is an array."
                )
            nbar_z = jnp.asarray(galaxies["nbar_z"], dtype=float)
            if nbar_z.shape != nbar.shape:
                raise ValueError(
                    "galaxies['nbar_z'] and galaxies['nbar'] differ in shape."
                )
            analysis["nbar_gal_comoving_zarray"] = nbar_z
            analysis["nbar_gal_comoving_val"] = nbar

    hod = galaxies.get("hod") or {}
    unknown = set(hod) - set(HOD_PARAMETERS)
    if unknown:
        raise ValueError(
            f"Unknown HOD parameters {sorted(unknown)}; expected {HOD_PARAMETERS}."
        )
    z_edges = galaxies.get("z_edges")
    if z_edges is not None:
        z_edges = jnp.asarray(z_edges, dtype=float)
        if z_edges.ndim != 2 or z_edges.shape[0] == 0 or z_edges.shape[1] != 2:
            raise ValueError(
                f"galaxies['z_edges'] must have shape (n_bins, 2), got {z_edges.shape}."
            )
        if not bool(jnp.all(z_edges[:, 0] < z_edges[:, 1])):
            raise ValueError(
                "galaxies['z_edges'] must have a lower edge below the upper edge "
                "in every bin."
            )

    # GODMAX reads per-bin arrays at min(bin index, length - 1), so a short array
    # would silently reuse its last entry: only length 1 or n_bins + 1 is allowed.
    per_bin = False
    for name, value in hod.items():
        value = jnp.asarray(value, dtype=float)
        if value.ndim > 1 or value.size == 0:
            raise ValueError(
                f"HOD parameter {name!r} must be a scalar or a non-empty 1D array, got "
                f"shape {value.shape}."
            )
        value = jnp.atleast_1d(value)
        if value.shape[0] > 1:
            per_bin = True
            if z_edges is None:
                raise ValueError("galaxies['z_edges'] is required for a per-bin HOD.")
            if value.shape[0] != z_edges.shape[0] + 1:
                raise ValueError(
                    f"HOD parameter {name!r} has {value.shape[0]} entries; a "
                    f"per-bin HOD with {z_edges.shape[0]} bins needs "
                    f"{z_edges.shape[0] + 1} (default value first, then one per bin)."
                )
        sim[f"{name}_array"] = value

    if z_edges is not None:
        n_bins = z_edges.shape[0]
        # GODMAX reads the bin edges from the lens n(z) block; the n(z) itself
        # (flat here) only enters its C_ell kernels, which are not built.
        lens: Dict[str, Any] = {
            "nbins_lens": n_bins,
            "z_array_lens": jnp.linspace(jnp.min(z_edges), jnp.max(z_edges), 8),
            "z_edges_bins_lens": z_edges,
        }
        lens.update({f"nz{i}": 1.0 for i in range(n_bins)})
        analysis["nz_lens_info_dict"] = lens
    analysis["hod_params_model"] = "perbin" if per_bin else "combined"
    return sim, analysis


def build_godmax_config(
    redshifts: jnp.ndarray,
    model_tSZ: bool,
    galaxies: Optional[Dict[str, Any]],
    godmax_config: Optional[Dict[str, Dict[str, Any]]],
    halo_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Assemble the four GODMAX dicts (sim, halo, analysis, other), without cosmology.

    Args:
        redshifts: Halo-model redshift nodes.
        model_tSZ: Whether to compute the Compton-y spectra.
        galaxies: Galaxy sample and HOD settings, or None for no galaxies.
        godmax_config: Optional overrides, keyed by GODMAX section
            (`sim_params`, `halo_params`, `analysis`, `other_params`).
        halo_params: Halo-model parameters by GODMAX name
            (see `SIM_PARAMETERS`, `OTHER_PARAMETERS`).

    Returns:
        The GODMAX `sim_params`, `halo_params`, `analysis` and `other_params` dicts.
    """
    godmax_config = godmax_config or {}
    unknown = set(godmax_config) - set(_CONFIG_SECTIONS)
    if unknown:
        raise ValueError(
            f"Unknown godmax_config sections {sorted(unknown)}; "
            f"expected {_CONFIG_SECTIONS}."
        )
    for section, owned in _OWNED_KEYS.items():
        for key in sorted(set(godmax_config.get(section, {})) & set(owned)):
            raise ValueError(
                f"godmax_config['{section}']['{key}'] is set by cloelib from "
                f"`{owned[key]}`; pass it there instead."
            )

    unknown = set(halo_params) - set(SIM_PARAMETERS) - set(OTHER_PARAMETERS)
    if unknown:
        raise ValueError(
            f"Unknown halo-model parameters {sorted(unknown)}; expected names from "
            f"{SIM_PARAMETERS + OTHER_PARAMETERS}."
        )

    sim = dict(godmax_config.get("sim_params", {}))
    halo = {**DEFAULT_HALO_PARAMS, **godmax_config.get("halo_params", {})}
    analysis = {**DEFAULT_ANALYSIS, **godmax_config.get("analysis", {})}
    other = dict(godmax_config.get("other_params", {}))

    sim.update({k: v for k, v in halo_params.items() if k in SIM_PARAMETERS})
    other.update({k: v for k, v in halo_params.items() if k in OTHER_PARAMETERS})

    halo["z_array"] = jnp.asarray(redshifts, dtype=float)
    # GODMAX reads an ell grid and a source n(z) at construction; neither is used
    # for the 3D spectra, so both are kept minimal unless given.
    if "ell_array" not in halo and "ellmin" not in halo:
        halo["ell_array"] = jnp.array([10.0, 100.0])
    analysis.setdefault(
        "nz_source_info_dict",
        {"nbins": 1, "z_array_source": jnp.array([0.1, 1.0]), "nz0": 1.0},
    )

    analysis["model_galaxies"] = False
    if galaxies is not None:
        sim_gal, analysis_gal = _galaxy_config(
            galaxies, zheng_hod=bool(analysis.get("Zheng_HOD", False))
        )
        sim.update(sim_gal)
        analysis.update(analysis_gal)
    analysis["model_tSZ"] = bool(model_tSZ)
    return sim, halo, analysis, other


def _check_flat_w0cdm(background) -> None:
    """Raise unless the background is flat w0CDM without massive neutrinos.

    The check runs in NumPy, so it also works inside `jax.jit`; a traced value
    (e.g. `mnu`, which `JAXBackground` stores as a JAX array) is not checked.
    """
    for name in ("Omega_k0", "wa", "mnu"):
        value = getattr(background, name)
        if isinstance(value, jax.core.Tracer):
            continue
        if np.any(np.asarray(value) != 0.0):
            raise ValueError(
                f"GODMAX is flat w0CDM without massive neutrinos; {name} must be "
                f"0, got {value}."
            )


def _run_godmax(background, linearperturbations, nonlinearperturbations, config):
    """Run GODMAX up to its 3D power spectra (no C_ell) on a cloelib cosmology.

    Returns:
        The GODMAX `get_Pkz` object.
    """
    try:
        from godmax.base_class import base_class
        from godmax.get_Pkzs import get_Pkz
        from godmax.get_radial_profiles import Profiles
        from godmax.helpers.cloelib_cosmology import CloelibCosmology
    except ImportError as e:
        raise ImportError(
            f"godmax (with its cloelib cosmology backend) could not be imported: {e}. "
            "See the README for how to install GODMAX."
        ) from e

    sim, halo, analysis, other = config
    analysis = dict(analysis)
    analysis["cosmology_backend"] = "cloelib"
    analysis["cosmology_obj"] = CloelibCosmology(
        background, linearperturbations, nonlinearperturbations
    )
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    return get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)


def _spectra_in_cloelib_units(pkz, h: float):
    """Convert the GODMAX total spectra to cloelib units.

    Each Compton-y leg carries one factor of h on top of the volume factor h^-3:
    GODMAX's y profile is per Mpc/h along the line of sight, so that the Limber
    kernel of y is W_y(z) = 1 / (1 + z) with distances in Mpc after conversion.

    Returns:
        A tuple `(k, spectra)`: k in 1/Mpc, and a dict keyed by field pair with
        arrays of shape `(n_z, n_k)` in Mpc^3 (times 1/Mpc per y leg).
    """
    spectra = {}
    for pair, (name, n_y) in _SPECTRA.items():
        if "y" in pair and not pkz.model_tSZ:
            continue
        if "g" in pair and not pkz.model_galaxies:
            continue
        spectra[pair] = jnp.asarray(getattr(pkz, name)).T * h ** (n_y - 3)
    return pkz.kPk_array * h, spectra


def _interp_zk(zs, ks, z_nodes, k_nodes, values):
    """Interpolate positive `values` of shape `(n_z, n_k)` in (z, ln k).

    The logarithm is interpolated linearly in ln k, then in z; outside the grid
    the edge value is held.
    """
    log_k_nodes = jnp.log(k_nodes)
    log_values = jnp.log(values)
    at_ks = jax.vmap(lambda row: jnp.interp(jnp.log(ks), log_k_nodes, row))(log_values)
    at_zs = jax.vmap(lambda col: jnp.interp(zs, z_nodes, col), in_axes=1, out_axes=1)(
        at_ks
    )
    return jnp.exp(at_zs)


class GODMAXNonLinearPerturbations:
    """Non-linear matter, Compton-y and HOD galaxy spectra from the GODMAX halo model.

    The matter power spectrum is cloelib's halofit times the GODMAX baryonic
    response P_DMB / P_NFW. The amplitude is set by `As` of the background:
    cloelib converts it to sigma8, and GODMAX is normalised to that same sigma8.

    GODMAX is flat w0CDM without massive neutrinos, so the background must have
    `Omega_k0 = 0`, `wa = 0` and `mnu = 0`.

    Attributes:
        z: Halo-model redshift nodes.
        k: Halo-model wavenumbers in 1/Mpc.
        Pk: Matter power spectrum on the nodes, shape `(n_z, n_k)`, in Mpc^3.
        spectra: Total spectra on the nodes in cloelib units, keyed by field pair
            (`("m", "m")`, `("m", "y")`, `("y", "y")`, `("g", "g")`, `("g", "m")`,
            `("g", "y")`).
        godmax: The GODMAX `get_Pkz` object (h-units), for diagnostics such as the
            1-halo and 2-halo terms.
    """

    # Without massive neutrinos cb is the total matter, so sqrt(P_cb P_mm) = P_mm.
    cb_cross_prescription = CB_CROSS_CORRELATED

    #: Intrinsic-alignment models the tracers may use with these spectra (read
    #: by `ShearTracer`). TATT's one-loop terms are built from the matter field
    #: and have no halo-model counterpart for galaxies or Compton-y.
    supported_ia_models = (None, "NLA")
    #: Galaxy bias models the tracers may use with these spectra (read by
    #: `PositionsTracer`): the HOD, or a linear bias times the matter spectra.
    supported_galaxy_bias_models = ("hod", "per_bin", "per_bin_int", "poly")

    def __init__(
        self,
        background: JAXBackground,
        linearperturbations: Optional[JAXLinearPerturbations],
        redshifts: jnp.ndarray,
        *,
        model_tSZ: bool = True,
        galaxies: Optional[Dict[str, Any]] = None,
        godmax_config: Optional[Dict[str, Dict[str, Any]]] = None,
        **halo_params: Any,
    ) -> None:
        """
        Initialise the halo-model perturbations and run GODMAX.

        Args:
            background: A `JAXBackground` instance.
            linearperturbations: A `JAXLinearPerturbations` instance built on
                `background`, or None to build one.
            redshifts: Increasing redshift nodes of the halo model. The spectra are
                interpolated between them, so they should cover the redshifts of
                interest densely (about 50 nodes up to z = 4).
            model_tSZ: Whether to compute the Compton-y spectra.
            galaxies: Galaxy sample for the HOD spectra, or None. Keys: `nbar`
                (comoving number density in (h/Mpc)^3, scalar or array), `nbar_z`
                (redshifts of an `nbar` array), `z_edges` (bin edges, shape
                `(n_bins, 2)`), `hod` (HOD parameters by GODMAX name, scalar or
                per-bin array of length `n_bins + 1`). A per-bin HOD varies with
                true redshift over the `z_edges` intervals; it is not a separate
                population per tomographic bin (see "Per-bin HOD" in the module
                docstring). With `godmax_config["analysis"]["Zheng_HOD"] = True`,
                `nbar` and `nbar_z` are not allowed: the threshold is set by
                `hod["log10Mmin_Ncen"]`.
            godmax_config: Overrides of the GODMAX settings, keyed by section
                (`sim_params`, `halo_params`, `analysis`, `other_params`).
                A GODMAX parameter file can be passed through
                `godmax_config_from_params`.
            **halo_params: Halo-model parameters by GODMAX name, e.g.
                `theta_ej_0`, `theta_co_0`, `mu_beta`, `log10_Mc0`, `alpha_nt`.
                Parameters not given take GODMAX's code defaults, which might
                differ from its ``params_default.yaml``. The same holds for HOD
                parameters missing from `galaxies["hod"]`.
        """
        if isinstance(self, BaryonBoostMixin):
            raise TypeError(
                "GODMAXNonLinearPerturbations already includes baryons; do not "
                "combine it with with_baryon_boost."
            )
        if not isinstance(background, JAXBackground):
            raise ValueError(
                "GODMAXNonLinearPerturbations requires a JAXBackground, got "
                f"{type(background).__name__}."
            )
        _check_flat_w0cdm(background)

        redshifts = jnp.atleast_1d(jnp.asarray(redshifts, dtype=float))
        if redshifts.ndim != 1 or redshifts.shape[0] < 2:
            raise ValueError("redshifts must be a 1D array of at least two nodes.")
        if not bool(jnp.all(jnp.diff(redshifts) > 0)):
            raise ValueError("redshifts must be strictly increasing.")

        if linearperturbations is None:
            linearperturbations = JAXLinearPerturbations(background, redshifts)
        elif linearperturbations.background is not background:
            raise ValueError("linearperturbations was built on a different background.")

        config = build_godmax_config(
            redshifts, model_tSZ, galaxies, godmax_config, halo_params
        )

        self.background = background
        self.linearperturbations = linearperturbations
        self.halofit = JAXNonLinearPerturbations(
            background, linearperturbations, redshifts
        )
        self.godmax = _run_godmax(background, linearperturbations, self.halofit, config)

        self.z = redshifts
        # Plain floats, so that check_redshift_coverage works under jax.jit.
        self._z_min = float(redshifts[0])
        self._z_max = float(redshifts[-1])
        self.k, self.spectra = _spectra_in_cloelib_units(self.godmax, background.h)
        self.Pk = self.spectra[("m", "m")]
        # P_DMB / P_NFW (1 when GODMAX runs with `model_matter='halofit'`).
        self._response = jnp.asarray(
            self.godmax.Pmm_tot_mat / self.godmax.phfit_kz_mat
        ).T

    def check_redshift_coverage(self, z) -> None:
        """Raise if `z` extends outside the halo-model redshift nodes.

        The spectra are tabulated on the nodes; outside them they would be
        extrapolated. Called by `AngularTwoPoint` on the Limber grid. The check
        runs in NumPy, so it also works inside `jax.jit`; a traced `z` is not
        checked (as in `check_nonzero_redshifts`).
        """
        if isinstance(z, jax.core.Tracer):
            return
        z = np.atleast_1d(np.asarray(z))
        tolerance = 1e-8 * max(1.0, self._z_max)
        if z.min() < self._z_min - tolerance or z.max() > self._z_max + tolerance:
            raise ValueError(
                f"The redshift grid [{z.min()}, {z.max()}] extends outside the "
                f"halo-model nodes [{self._z_min}, {self._z_max}]; build "
                "GODMAXNonLinearPerturbations with redshifts covering it."
            )

    def growth_factor(self, zs, ks=None):
        """Return the linear growth factor D(z) / D(0)."""
        return self.linearperturbations.growth_factor(zs, ks)

    def growth_rate(self, zs=None, ks=None):
        """Return the linear growth rate (at the halo-model nodes if `zs` is None)."""
        return self.linearperturbations.growth_rate(self.z if zs is None else zs, ks)

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""
        return self.linearperturbations.sigma8_0()

    def baryonic_response(self, zs: jnp.ndarray, ks: jnp.ndarray) -> jnp.ndarray:
        """Return P_DMB / P_NFW of the halo model.

        Args:
            zs: Redshifts.
            ks: Wavenumbers in 1/Mpc.

        Returns:
            The response, shape `(len(zs), len(ks))`. Outside the halo-model grid
            the edge value is held.
        """
        zs, ks = jnp.atleast_1d(zs), jnp.atleast_1d(ks)
        return _interp_zk(zs, ks, self.z, self.k, self._response)

    def matter_power_spectrum(self, zs: jnp.ndarray, ks: jnp.ndarray) -> jnp.ndarray:
        """Compute the non-linear matter power spectrum including baryons.

        Args:
            zs: Redshifts.
            ks: Wavenumbers in 1/Mpc.

        Returns:
            The power spectrum in Mpc^3, shape `(len(zs), len(ks))`: cloelib's
            halofit times the halo-model baryonic response.
        """
        zs, ks = jnp.atleast_1d(zs), jnp.atleast_1d(ks)
        halofit = self.halofit.matter_power_spectrum(zs, ks)
        return jnp.reshape(halofit, (len(zs), len(ks))) * self.baryonic_response(zs, ks)

    def matter_power_spectrum_cb(self, zs: jnp.ndarray, ks: jnp.ndarray) -> jnp.ndarray:
        """Compute the cold dark matter + baryons spectrum.

        GODMAX has no massive neutrinos (`mnu = 0` is enforced), so this is the
        total matter spectrum, `matter_power_spectrum`.
        """
        return self.matter_power_spectrum(zs, ks)

    def power_spectrum(
        self, field1: str, field2: str, zs: jnp.ndarray, ks: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute a halo-model power spectrum of two fields.

        Fields are `"m"` (matter), `"g"` (HOD galaxies) and `"y"` (Compton-y). The
        y spectra are defined for a Limber kernel `W_y(z) = 1 / (1 + z)` with
        distances in Mpc.

        Args:
            field1: First field.
            field2: Second field.
            zs: Redshifts.
            ks: Wavenumbers in 1/Mpc.

        Returns:
            The spectrum, shape `(len(zs), len(ks))`, in Mpc^3 (times 1/Mpc per y
            leg). Matter-matter is `matter_power_spectrum`; the others are
            interpolated in (z, ln k) and held at the edge value outside the grid.
            Check that the k grid (`self.k`, set by the halo-model `kmax` in
            h/Mpc) covers (ell + 1/2) / chi(z_min) for the largest ell.
        """
        pair = tuple(sorted((field1, field2)))
        if pair == ("m", "m"):
            return self.matter_power_spectrum(zs, ks)
        if pair not in self.spectra:
            if set(pair) <= {"m", "g", "y"}:
                needed = []
                if "g" in pair and ("g", "g") not in self.spectra:
                    needed.append("galaxies=...")
                if "y" in pair and ("y", "y") not in self.spectra:
                    needed.append("model_tSZ=True")
                missing = " and ".join(needed)
                raise ValueError(
                    f"The {field1}-{field2} spectrum was not computed; construct "
                    f"GODMAXNonLinearPerturbations with {missing}."
                )
            raise ValueError(
                f"Unknown fields {field1!r}, {field2!r}; expected 'm', 'g' or 'y'."
            )
        zs, ks = jnp.atleast_1d(zs), jnp.atleast_1d(ks)
        return _interp_zk(zs, ks, self.z, self.k, self.spectra[pair])
