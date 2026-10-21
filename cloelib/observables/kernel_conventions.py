"""
Kernel conventions for analyses on halo-model spectra (GODMAX).

With `GODMAXNonLinearPerturbations`, every pair of fields gets its halo-model
spectrum; what remains to choose is which systematics the tracer kernels
include. Two presets:

- ``"godmax"``: the kernels as GODMAX builds them. HOD galaxies, NLA with a
  pivot redshift, no magnification, no RSD, no galaxy photo-z shift or n(z)
  width, CMB lensing source at z = 1100, convergence (no spin-2 factor), and
  GODMAX's default beam on the y map.
- ``"baseline"``: the cloelib kernels. Linear galaxy bias times the matter
  spectra, NLA, magnification, RSD, photo-z shift and n(z) width for every
  tracer, CMB lensing source at the background's z_star, spin-2 factor, no
  beam.

Any ingredient can be overridden, e.g. ``kernel_convention("godmax",
magnification=True)``. TATT and non-linear galaxy bias are not available on
the halo-model route.

Usage::

    convention = kernel_convention("godmax", rsd=True)
    nuisance = convention.nuisance(nuisance_params)
    shear = ShearTracer(perturbations, nz_s, z, nuisance, **convention.shear_options())
    galaxies = PositionsTracer(
        perturbations, nz_g, z, nuisance_params=nuisance,
        **convention.positions_options(),
    )
    cmb = CMBLensingTracer(perturbations, z, **convention.cmb_options())
    tsz = tSZTracer(perturbations, z, **convention.tsz_options())
"""

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional

#: Linear galaxy bias models of `PositionsTracer`.
LINEAR_BIAS_MODELS = ("per_bin", "per_bin_int", "poly")
#: Galaxy bias models available on the halo-model route.
GALAXY_BIAS_MODELS = ("hod",) + LINEAR_BIAS_MODELS
#: Intrinsic-alignment choices: none, cloelib's NLA, or NLA with GODMAX's
#: pivot redshift, ((1+z)/(1+z0))**eta.
IA_MODELS = (None, "NLA", "NLA_pivot")
#: CMB lensing source redshift used by GODMAX.
GODMAX_CMB_SOURCE_REDSHIFT = 1100.0
#: Default FWHM of the y-map beam in GODMAX, in arcmin.
GODMAX_BEAM_FWHM_ARCMIN = 1.4
#: Default NLA pivot redshift in GODMAX.
GODMAX_IA_PIVOT_REDSHIFT = 0.62


@dataclass(frozen=True)
class KernelConvention:
    """Which systematics the tracer kernels include.

    Attributes:
        galaxy_bias_model: `"hod"` (bias from the halo model, in the spectra)
            or a linear model of `PositionsTracer` (`"per_bin"`,
            `"per_bin_int"`, `"poly"`; bias in the kernel, matter spectra).
        ia_model: `None` (no IA), `"NLA"` (cloelib, (1+z)**EtaIA) or
            `"NLA_pivot"` (GODMAX, ((1+z)/(1+z0IA))**EtaIA).
        magnification: Whether the galaxy kernel has the magnification term.
        rsd: Whether the galaxy kernel has the photometric RSD term.
        galaxy_photoz: Whether the galaxy n(z) is shifted (`dz_pos_i`) and
            stretched (`width_pos_i`).
        shear_nz_width: Whether the shear n(z) is stretched (`width_shear_i`).
            The shear shift `dz_shear_i` and multiplicative bias are in both
            codes and always kept.
        cmb_z_source: CMB lensing source redshift, or None for the
            background's z_star.
        spin2_prefactor: Whether shear legs carry the spin-2 ell factor
            (shear E-mode spectrum) or not (convergence, as in GODMAX).
        y_beam_fwhm_arcmin: FWHM of the Gaussian beam of the y map, in arcmin
            (0 for none).
    """

    galaxy_bias_model: str
    ia_model: Optional[str]
    magnification: bool
    rsd: bool
    galaxy_photoz: bool
    shear_nz_width: bool
    cmb_z_source: Optional[float]
    spin2_prefactor: bool
    y_beam_fwhm_arcmin: float

    def __post_init__(self) -> None:
        if self.galaxy_bias_model == "nonlinear":
            raise NotImplementedError(
                "galaxy_bias_model='nonlinear' is not available on the halo-model "
                f"route; use one of {GALAXY_BIAS_MODELS}."
            )
        if self.galaxy_bias_model not in GALAXY_BIAS_MODELS:
            raise ValueError(
                f"Unknown galaxy_bias_model {self.galaxy_bias_model!r}; expected "
                f"one of {GALAXY_BIAS_MODELS}."
            )
        if self.ia_model == "TATT":
            raise NotImplementedError(
                "ia_model='TATT' is not available on the halo-model route; use "
                f"one of {IA_MODELS}."
            )
        if self.ia_model not in IA_MODELS:
            raise ValueError(
                f"Unknown ia_model {self.ia_model!r}; expected one of {IA_MODELS}."
            )
        if self.y_beam_fwhm_arcmin < 0:
            raise ValueError("y_beam_fwhm_arcmin must be non-negative.")

    def nuisance(self, nuisance_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the nuisance parameters for this convention.

        Systematics that are switched off get their neutral value
        (`magnification_bias_i = 0`, `dz_pos_i = 0`, `width_pos_i = 1`,
        `width_shear_i = 1`). With `ia_model="NLA_pivot"`, `AIA` is rescaled to
        `AIA * (1 + z0IA)**(-EtaIA)`, which turns cloelib's NLA into GODMAX's
        exactly; `z0IA` defaults to GODMAX's 0.62.

        The shear shift is kept as given. Note the sign conventions: cloelib's
        `dz_shear_i` gives n(z + dz), GODMAX's `Delta_z_bias` n(z - dz), so a
        GODMAX value maps to `dz_shear_i = -Delta_z_bias`.
        """
        params = dict(nuisance_params)
        for key in params:
            if not self.magnification and key.startswith("magnification_bias_"):
                params[key] = 0.0
            if not self.galaxy_photoz and key.startswith("dz_pos_"):
                params[key] = 0.0
            if not self.galaxy_photoz and key.startswith("width_pos_"):
                params[key] = 1.0
            if not self.shear_nz_width and key.startswith("width_shear_"):
                params[key] = 1.0
        if self.ia_model == "NLA_pivot":
            z0 = params.get("z0IA", GODMAX_IA_PIVOT_REDSHIFT)
            params["AIA"] = params["AIA"] * (1.0 + z0) ** (-params["EtaIA"])
        return params

    def shear_options(self) -> Dict[str, Any]:
        """Keyword arguments for `ShearTracer`."""
        return {
            "ia_model": None if self.ia_model is None else "NLA",
            "spin2_prefactor": self.spin2_prefactor,
        }

    def positions_options(self) -> Dict[str, Any]:
        """Keyword arguments for `PositionsTracer`."""
        return {"galaxy_bias_model": self.galaxy_bias_model, "include_rsd": self.rsd}

    def cmb_options(self) -> Dict[str, Any]:
        """Keyword arguments for `CMBLensingTracer`."""
        return {"z_source": self.cmb_z_source}

    def tsz_options(self) -> Dict[str, Any]:
        """Keyword arguments for `tSZTracer`."""
        return {"beam_fwhm_arcmin": self.y_beam_fwhm_arcmin}


PRESETS = {
    "godmax": KernelConvention(
        galaxy_bias_model="hod",
        ia_model="NLA_pivot",
        magnification=False,
        rsd=False,
        galaxy_photoz=False,
        shear_nz_width=False,
        cmb_z_source=GODMAX_CMB_SOURCE_REDSHIFT,
        spin2_prefactor=False,
        y_beam_fwhm_arcmin=GODMAX_BEAM_FWHM_ARCMIN,
    ),
    "baseline": KernelConvention(
        galaxy_bias_model="poly",
        ia_model="NLA",
        magnification=True,
        rsd=True,
        galaxy_photoz=True,
        shear_nz_width=True,
        cmb_z_source=None,
        spin2_prefactor=True,
        y_beam_fwhm_arcmin=0.0,
    ),
}


def kernel_convention(preset: str = "godmax", **overrides) -> KernelConvention:
    """Return the kernel convention of a preset, with optional overrides.

    Args:
        preset: `"godmax"` or `"baseline"` (see the module docstring).
        **overrides: Any field of `KernelConvention`, e.g. `rsd=True`,
            `galaxy_bias_model="per_bin"`, `ia_model=None`.

    Returns:
        The `KernelConvention`.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {list(PRESETS)}.")
    names = {f.name for f in fields(KernelConvention)}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(
            f"Unknown kernel-convention options {sorted(unknown)}; expected names "
            f"from {sorted(names)}."
        )
    return replace(PRESETS[preset], **overrides)
