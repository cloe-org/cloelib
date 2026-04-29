"""Linear perturbation wrapper for beyond-LCDM growth models via MGrowth."""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from cloelib.cosmology.cosmology import Background, Perturbations

try:
    import MGrowth as mgrowth
except ImportError as exc:
    raise ImportError("MGrowth could not be imported or initialised.") from exc


class MGrowthLinearPerturbations:
    """Augment a baseline linear perturbation object with MGrowth corrections."""

    def __init__(
        self,
        background: Background,
        base_linear_perturbations: Perturbations,
        gravity_model: str,
        mgpars: dict,
    ) -> None:
        """Initialise beyond-LCDM linear perturbations using MGrowth.

        Parameters
        ----------
        background : Background
            Background object for the beyond-LCDM cosmology.
        base_linear_perturbations : Perturbations
            Baseline linear perturbations, typically LCDM or w0waCDM.
        gravity_model : str
            MGrowth model tag. Supported values are `w0wacdm`, `fr`, `dgp`,
            `ide`, `gamma`, `gammaz`, `musigma-de`, and `mu`.
        mgpars : dict
            Additional model parameters required by the chosen MGrowth model.
        """
        if background.Omega_k0 != 0:
            raise ValueError("Non flat geometries not supported")

        self.background = background
        self.base = base_linear_perturbations
        self.gravity_model = gravity_model.lower()
        self.mgpars = mgpars

        self.z = self.base.z
        self.z_sorted = np.linspace(0.0, 5.0, 256, endpoint=True)
        self.a_interp = np.linspace(1e-3, 1.0, 128)
        self.a_sorted = (1.0 / (1.0 + self.z_sorted))[::-1]

        # Use a reduced k-grid for MGrowth evaluations and interpolate later.
        self.k = np.logspace(
            np.log10(self.base.k[0]),
            np.log10(self.base.k[-1]),
            128,
            endpoint=True,
        )
        self.k_len = len(self.k)

        background_mgrowth = {
            "Omega_m": self.background.Omega_m(np.array([0.0]))[0],
            "h": self.background.h,
            "w0": getattr(self.background, "w0", -1.0),
            "wa": getattr(self.background, "wa", 0.0),
            "a_arr": self.a_sorted,
        }

        mg_models = {
            "w0wacdm": mgrowth.w0waCDM,
            "ide": mgrowth.IDE,
            "fr": mgrowth.fR_HS,
            "dgp": mgrowth.nDGP,
            "gamma": mgrowth.Linder_gamma,
            "gammaz": mgrowth.Linder_gamma_a,
            "musigma-de": mgrowth.mu_a,
            "mu": mgrowth.mu_a,
        }
        if self.gravity_model not in mg_models:
            raise ValueError(f"Unsupported gravity model '{self.gravity_model}'.")

        self.check_ranges = True
        if self.gravity_model in {"musigma-de", "mu"}:
            if "mu0" not in mgpars or "sigma0" not in mgpars:
                raise ValueError("Mu-Sigma parameters are not properly specified.")
            self.mu_interp = self._compute_mu_de_interp(
                mgpars["mu0"],
                background_mgrowth["Omega_m"],
                background_mgrowth["w0"],
                background_mgrowth["wa"],
            )
            self.sigma_lensing = self._compute_sigma_de_interp(
                mgpars["sigma0"],
                background_mgrowth["Omega_m"],
                background_mgrowth["w0"],
                background_mgrowth["wa"],
            )
            if mgpars["mu0"] > 2.0 * mgpars["sigma0"] + 1.0:
                self.check_ranges = False

        self.mg_cosmo = mg_models[self.gravity_model](background_mgrowth)
        if self.gravity_model in {"musigma-de", "mu"}:
            self._compute_growth_generic = self._compute_growth_muinterp
        else:
            self._compute_growth_generic = getattr(
                self, f"_compute_growth_{self.gravity_model}", None
            )
        if self._compute_growth_generic is None:
            raise ValueError(
                f"Growth computation for model '{self.gravity_model}' is not implemented."
            )

        self._compute_growth(background_mgrowth)

    def _compute_mu_de_interp(self, mu0, omega0, w0, wa):
        omega_l = (
            (1.0 - omega0)
            * self.a_interp ** (-3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * (1.0 - self.a_interp))
        )
        omega_l0 = 1.0 - omega0
        e2 = omega0 / self.a_interp**3 + omega_l
        mu_de = 1.0 + mu0 * (omega_l / e2) / omega_l0
        return interpolate.interp1d(
            self.a_interp,
            mu_de,
            bounds_error=False,
            kind="cubic",
            fill_value=(mu_de[0], mu_de[-1]),
        )

    def _compute_sigma_de_interp(self, sigma0, omega0, w0, wa):
        omega_l = (
            (1.0 - omega0)
            * (1.0 + self.z_sorted) ** (3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * self.z_sorted / (1.0 + self.z_sorted))
        )
        omega_l0 = 1.0 - omega0
        e2 = omega0 * (1.0 + self.z_sorted) ** 3 + omega_l
        sigma_de = 1.0 + sigma0 * (omega_l / e2) / omega_l0
        return interpolate.interp1d(
            self.z_sorted,
            sigma_de,
            bounds_error=False,
            kind="cubic",
            fill_value=(sigma_de[0], sigma_de[-1]),
        )

    def _repeat_scale_independent_growth(self, d_raw, f_raw):
        d_arr = np.repeat(np.asarray(d_raw)[::-1, None], self.k_len, axis=1)
        f_arr = np.repeat(np.asarray(f_raw)[::-1, None], self.k_len, axis=1)
        return d_arr, f_arr

    def _compute_growth_w0wacdm(self):
        return self._repeat_scale_independent_growth(*self.mg_cosmo.growth_parameters())

    def _compute_growth_dgp(self):
        return self._repeat_scale_independent_growth(
            *self.mg_cosmo.growth_parameters(omegarc=self.mgpars["omega_rc"])
        )

    def _compute_growth_fr(self):
        d_raw, f_raw = self.mg_cosmo.growth_parameters(
            k_arr=self.k * self.background.h,
            fR0=self.mgpars["fr0"],
        )
        return np.asarray(d_raw).T[::-1, :], np.asarray(f_raw).T[::-1, :]

    def _compute_growth_ide(self):
        h = self.background.H0 / 100.0
        omega_cdm = self.background.Omega_cdm0
        omega0 = (
            omega_cdm + self.background.Omega_b0 + self.background.mnu / 93.14 / h**2
        )
        rc = omega_cdm / h**2 / omega0
        unit_conv = 0.0194407
        xi = self.mgpars["xi"]
        w0 = self.background.w0
        xi_corrected = (
            xi
            * rc
            / (1.0 + unit_conv * h * (1.0 - omega0) * (1.0 + w0) * xi * (1.0 - rc))
        )
        return self._repeat_scale_independent_growth(
            *self.mg_cosmo.growth_parameters(xi=xi_corrected)
        )

    def _compute_growth_gamma(self):
        return self._repeat_scale_independent_growth(
            *self.mg_cosmo.growth_parameters(gamma=self.mgpars["gamma0"])
        )

    def _compute_growth_gammaz(self):
        return self._repeat_scale_independent_growth(
            *self.mg_cosmo.growth_parameters(
                gamma0=self.mgpars["gamma0"], gamma1=self.mgpars["gamma1"]
            )
        )

    def _compute_growth_muinterp(self):
        return self._repeat_scale_independent_growth(
            *self.mg_cosmo.growth_parameters(mu_interp=self.mu_interp)
        )

    def _compute_growth(self, bg_dict):
        d_mg, f_mg = self._compute_growth_generic()

        base_cosmo = mgrowth.w0waCDM(bg_dict)
        d_base_raw, _ = base_cosmo.growth_parameters()
        d_base = np.asarray(d_base_raw)[::-1]

        self.dz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, d_mg, kx=1, ky=1
        )
        self.dz_norm_dz0_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, d_mg / d_mg[0, :], kx=1, ky=1
        )
        self.fz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, f_mg, kx=1, ky=1
        )
        self.dz_norm_lcdm_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, d_mg / d_base[0, None], kx=1, ky=1
        )

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Calculate the growth factor D(z, k) normalised to D(0)."""
        return self.dz_norm_dz0_interp(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """Calculate the growth rate f(z, k)."""
        return self.fz_interp(zs, ks)

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Compute the beyond-LCDM linear matter power spectrum."""
        ps_base = self.base.matter_power_spectrum(0.0, ks)
        return self.dz_norm_lcdm_interp(zs, ks) ** 2 * ps_base

    def sigma8_0(self) -> float:
        """Approximate sigma8 today from the baseline value and growth rescaling."""
        k_ref = float(np.min(self.k))
        d_ratio = float(
            self.dz_norm_lcdm_interp(np.array([0.0]), np.array([k_ref]))[0, 0]
        )
        return float(self.base.sigma8_0()) * d_ratio
