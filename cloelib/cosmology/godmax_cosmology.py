"""
Implementation of the GODMAX halo model on top of cloelib's JAX cosmology.

## Design

**``GODMAXNonLinearPerturbations``** runs the GODMAX halo model on cloelib's
JAX background, Eisenstein-Hu linear P(k) and halofit, and provides:

- the non-linear matter power spectrum: cloelib's halofit times the GODMAX
  baryonic response P_DMB / P_NFW;
- the halo-model spectra of matter (``"m"``), Compton-y (``"y"``, tSZ) and HOD
  galaxies (``"g"``), through ``power_spectrum(field1, field2, zs, ks)``. These
  are used by `cloelib.observables.tsz.tSZTracer` and by
  `PositionsTracer(..., galaxy_bias_model="hod")`.

## Units

GODMAX works in h-units (k in h/Mpc, P in (Mpc/h)^3, distances in Mpc/h);
cloelib works in Mpc (k in 1/Mpc, P in Mpc^3). The conversion is done in this
module only (`_spectra_in_cloelib_units`).

## Notes

- GODMAX is not on PyPI. Install it from GitHub (see the README); it needs the
  ``godmax.helpers.cloelib_cosmology`` module.
- GODMAX is flat w0CDM without massive neutrinos.
- `DEFAULT_HALO_PARAMS` and `DEFAULT_ANALYSIS` mirror GODMAX's
  ``params_default.yaml``; they do not follow changes there automatically.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from cloelib.cosmology.cosmology import CB_CROSS_CORRELATED
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)

#: Halo-model grids and models (the settings of GODMAX's `params_default.yaml`).
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
    "mdef_Delta": 200,
    "conc_model": "Duffy08",
    "hmf_model": "T10",
    "do_corr_2h_mm": True,
}

#: Analysis switches and accuracy settings (the settings of GODMAX's
#: `params_default.yaml`). `model_tSZ`, `model_galaxies` and the cosmology are
#: always set by the perturbations class.
DEFAULT_ANALYSIS: Dict[str, Any] = {
    "backreaction": True,
    "baryonification_tSZ": True,
    "num_points_trapz_int": 32,
    "num_points_gal_cal": 32,
    "lowpass_Pmm1h_lowk": True,
    "kthresh_lowpass_Pmm1h_lowk": 1e-3,
    "split_cga_fftlog": True,
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
#: transition smoothing of each spectrum).
OTHER_PARAMETERS = ("alpha_ky", "alpha_gy", "alpha_gg", "alpha_gm", "alpha_ge")

#: HOD parameters accepted in `galaxies["hod"]`; each is a scalar or, for a
#: per-bin HOD, an array of length n_bins + 1 (default value first).
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
)

_GALAXIES_KEYS = ("nbar", "nbar_z", "z_edges", "hod")
_CONFIG_SECTIONS = ("sim_params", "halo_params", "analysis", "other_params")

#: (field1, field2) -> (GODMAX attribute, number of Compton-y legs).
_SPECTRA = {
    ("m", "m"): ("Pmm_tot_mat", 0),
    ("m", "y"): ("Pym_tot_mat", 1),
    ("y", "y"): ("Pyy_tot_kz_mat", 2),
    ("g", "g"): ("Pgg_tot_mat", 0),
    ("g", "m"): ("Pgm_tot_mat", 0),
    ("g", "y"): ("Pgy_tot_mat", 1),
}


def _galaxy_config(galaxies: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate the `galaxies` dict into GODMAX `sim_params` and `analysis` entries."""
    unknown = set(galaxies) - set(_GALAXIES_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown galaxies keys {sorted(unknown)}; expected {_GALAXIES_KEYS}."
        )
    if "nbar" not in galaxies:
        raise ValueError(
            "galaxies['nbar'] (comoving number density in (h/Mpc)^3) is required: "
            "GODMAX sets the stellar-mass threshold of the HOD from it."
        )

    sim: Dict[str, Any] = {}
    analysis: Dict[str, Any] = {"model_galaxies": True}

    nbar = jnp.asarray(galaxies["nbar"], dtype=float)
    if nbar.ndim == 0:
        analysis["nbar_gal_comoving_val"] = float(nbar)
    else:
        if galaxies.get("nbar_z") is None:
            raise ValueError("galaxies['nbar_z'] is required when nbar is an array.")
        nbar_z = jnp.asarray(galaxies["nbar_z"], dtype=float)
        if nbar_z.shape != nbar.shape:
            raise ValueError("galaxies['nbar_z'] and galaxies['nbar'] differ in shape.")
        analysis["nbar_gal_comoving_zarray"] = nbar_z
        analysis["nbar_gal_comoving_val"] = nbar

    hod = galaxies.get("hod") or {}
    unknown = set(hod) - set(HOD_PARAMETERS)
    if unknown:
        raise ValueError(
            f"Unknown HOD parameters {sorted(unknown)}; expected {HOD_PARAMETERS}."
        )
    per_bin = False
    for name, value in hod.items():
        value = jnp.atleast_1d(jnp.asarray(value, dtype=float))
        per_bin = per_bin or value.shape[0] > 1
        sim[f"{name}_array"] = value

    z_edges = galaxies.get("z_edges")
    if z_edges is None:
        if per_bin:
            raise ValueError("galaxies['z_edges'] is required for a per-bin HOD.")
    else:
        z_edges = jnp.asarray(z_edges, dtype=float).reshape(-1, 2)
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
    redshifts,
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
    if "cosmo" in godmax_config.get("sim_params", {}):
        raise ValueError(
            "The cosmology is taken from the cloelib background; "
            "godmax_config['sim_params']['cosmo'] is not allowed."
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
        sim_gal, analysis_gal = _galaxy_config(galaxies)
        sim.update(sim_gal)
        analysis.update(analysis_gal)
    # GODMAX needs the pressure profiles for its galaxy spectra.
    analysis["model_tSZ"] = bool(model_tSZ) or galaxies is not None
    return sim, halo, analysis, other


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
        redshifts,
        *,
        model_tSZ: bool = True,
        galaxies: Optional[Dict[str, Any]] = None,
        godmax_config: Optional[Dict[str, Dict[str, Any]]] = None,
        **halo_params,
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
                per-bin array of length `n_bins + 1`).
            godmax_config: Overrides of the GODMAX settings, keyed by section
                (`sim_params`, `halo_params`, `analysis`, `other_params`).
            **halo_params: Halo-model parameters by GODMAX name, e.g.
                `theta_ej_0`, `theta_co_0`, `mu_beta`, `log10_Mc0`, `alpha_nt`.
                Parameters not given take GODMAX's defaults.
        """
        if not isinstance(background, JAXBackground):
            raise ValueError(
                "GODMAXNonLinearPerturbations requires a JAXBackground, got "
                f"{type(background).__name__}."
            )
        for name in ("Omega_k0", "wa", "mnu"):
            if float(getattr(background, name)) != 0.0:
                raise ValueError(
                    f"GODMAX is flat w0CDM without massive neutrinos; {name} must be "
                    f"0, got {getattr(background, name)}."
                )

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
        self.k, self.spectra = _spectra_in_cloelib_units(self.godmax, background.h)
        self.Pk = self.spectra[("m", "m")]
        # P_DMB / P_NFW (1 when GODMAX runs with `model_matter='halofit'`).
        self._response = jnp.asarray(
            self.godmax.Pmm_tot_mat / self.godmax.phfit_kz_mat
        ).T

    def check_redshift_coverage(self, z) -> None:
        """Raise if `z` extends outside the halo-model redshift nodes.

        The spectra are tabulated on the nodes; outside them they would be
        extrapolated. Called by `AngularTwoPoint` on the Limber grid.
        """
        z = jnp.atleast_1d(jnp.asarray(z))
        z_min, z_max = float(self.z[0]), float(self.z[-1])
        tolerance = 1e-8 * max(1.0, z_max)
        if (
            float(jnp.min(z)) < z_min - tolerance
            or float(jnp.max(z)) > z_max + tolerance
        ):
            raise ValueError(
                f"The redshift grid [{float(jnp.min(z))}, {float(jnp.max(z))}] "
                f"extends outside the halo-model nodes [{z_min}, {z_max}]; build "
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

    def baryonic_response(self, zs, ks):
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

    def matter_power_spectrum(self, zs, ks):
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

    def matter_power_spectrum_cb(self, zs, ks):
        """Compute the cold dark matter + baryons spectrum.

        GODMAX has no massive neutrinos (`mnu = 0` is enforced), so this is the
        total matter spectrum, `matter_power_spectrum`.
        """
        return self.matter_power_spectrum(zs, ks)

    def power_spectrum(self, field1: str, field2: str, zs, ks):
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
        """
        pair = tuple(sorted((field1, field2)))
        if pair == ("m", "m"):
            return self.matter_power_spectrum(zs, ks)
        if pair not in self.spectra:
            if set(pair) <= {"m", "g", "y"}:
                missing = "galaxies=..." if "g" in pair else "model_tSZ=True"
                raise ValueError(
                    f"The {field1}-{field2} spectrum was not computed; construct "
                    f"GODMAXNonLinearPerturbations with {missing}."
                )
            raise ValueError(
                f"Unknown fields {field1!r}, {field2!r}; expected 'm', 'g' or 'y'."
            )
        zs, ks = jnp.atleast_1d(zs), jnp.atleast_1d(ks)
        return _interp_zk(zs, ks, self.z, self.k, self.spectra[pair])
