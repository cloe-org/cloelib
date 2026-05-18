"""Implementation of Linear Perturbation in extended cosmologies."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations

from scipy import interpolate, integrate

# General imports
import numpy as np

# Cosmology imports
try:
    import MGrowth as mgrowth
except ImportError:
    raise ImportError("MGrowth could not be imported or initialised.")


# Notes:
# - Adapted from ABC classes


class MGrowthLinearPerturbations:
    """Class for linear perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        base_linear_perturbations: Perturbations,
        gravity_model: str,
        mgpars: dict,
    ):
        """Initialize the MGLinearPerturbations class to compute linear growth in modified gravity (MG).

        This class augments a standard LCDM or w0waCDM linear perturbation object with
        modifications to the growth factor and power spectrum using the MGrowth
        package, enabling f(R), DGP, ds, and other parameterized gravity models.

        Parameters
        ----------
        background : Background
            A cosmological background instance containing parameters such as
            Omega_m, h, w0, and wa.

        base_linear_perturbations : Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the LCDM or w0waCDM baseline.

        gravity_model : str
            Name of the MGrowth-supported gravity model to use.
            Examples: 'w0wacdm', 'fr', 'dgp', 'ds', 'gamma', 'gammaz', 'musigma-de', 'mu'.

        mgpars : dictionary of the extended parameters
            Examples: fR0 for f(R), omegarc for DGP, gamma0/gamma1 for Linder models, mu0/Sigma0 or binned values for mu-Sigma, xi for ds,
            screening parameters etc.

        Notes
        -----
        This class ensures:
        - Redshifts are reversed to match MGrowth's expected ascending scale factors.
        - An interpolator for the gravitational potential modification is computed.
        - If the model is scale-dependent (like f(R)), the returned growth factor D(z, k)
        is reshaped and ordered to match the base perturbation object's (z, k) convention.
        - The w0waCDM growth used for rescaling is also computed using MGrowth to maintain
        internal consistency.
        """
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        # must match the background in the extended cosmology
        self.background = background
        # must be w0waCDM (i.e. or wCDM, or LCDM)
        self.base = base_linear_perturbations

        # Sort scale factors and redshifts in ascending order (early to late times)
        # hardcoded
        self.z_sorted = np.linspace(0.0, 5.0, 256, endpoint=True)
        # Get scale factors
        self.a = 1.0 / (1.0 + self.z_sorted)
        self.a_sorted = self.a[::-1]

        # hardcoded to save time for f(R) by setting 128 k-values and interpolating later
        self.k = np.logspace(
            np.log10(self.base.k[0]), np.log10(self.base.k[-1]), 128, endpoint=True
        )  # in 1/Mpc
        self.k_len = len(self.k)

        self.gravity_model = gravity_model.lower()
        self.mgpars = mgpars

        # Build background dict for MGrowth
        background_mgrowth = {
            "Omega_m": self.background.Omega_m(np.array([0.0]))[0],
            "h": self.background.h,
            "w0": getattr(self.background, "w0", -1.0),
            "wa": getattr(self.background, "wa", 0.0),
            "a_arr": self.a_sorted,
        }

        # Available MGrowth models (see https://github.com/MariaTsedrik/MGrowth)
        mg_models = {
            "w0wacdm": mgrowth.w0waCDM,
            "ds": mgrowth.IDE,
            "fr": mgrowth.fR_HS,
            "dgp": mgrowth.nDGP,
            "gamma": mgrowth.Linder_gamma,
            "gammaz": mgrowth.Linder_gamma_a,
            "musigma-de": mgrowth.mu_a,
            "mu": mgrowth.mu_a,  # similar notation to ReACTemu_cosmology
        }
        if self.gravity_model not in mg_models:
            raise ValueError(f"Unsupported gravity model '{self.gravity_model}'.")

        # Flag to control model-specific priors:
        self.check_ranges = True

        # If musigma-de: specify mu-interp as a function of dark energy evolution;
        # for now only CPL-like dark energy models are supported but can be easily extended.
        self.a_interp = np.linspace(1e-3, 1.0, 128)
        if self.gravity_model == "musigma-de":
            if "mu0" in mgpars and "sigma0" in mgpars:
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
                # If mu0 > 2*sigma0+1, MGCAMB breaks but MGrowth does not;
                # so it depends on your setup, by default this condition
                # is imposed because nonlinear emulators used MGCAMB.
                if mgpars["mu0"] > 2.0 * mgpars["sigma0"] + 1.0:
                    self.check_ranges = False
            else:
                raise ValueError("Mu-Sigma parameters are not properly specified.")
        if self.gravity_model == "mu":
            if (
                "mu0" in mgpars
                and "sigma0" in mgpars
                and "c1" in mgpars
                and "c2" in mgpars
                and "lam" in mgpars
            ):
                self.mu_interp = self._compute_mu_de_scaledep_interp(
                    mgpars["mu0"],
                    mgpars["c1"],
                    mgpars["lam"],
                    background_mgrowth["Omega_m"],
                    background_mgrowth["w0"],
                    background_mgrowth["wa"],
                )
                self.sigma_lensing = self._compute_sigma_de_scaledep_interp(
                    mgpars["sigma0"],
                    mgpars["c2"],
                    mgpars["lam"],
                    background_mgrowth["Omega_m"],
                    background_mgrowth["w0"],
                    background_mgrowth["wa"],
                )
                # If mu0 > 2*sigma0+1, MGCAMB breaks but MGrowth does not;
                # so it depends on your setup, by default this condition
                # is imposed because nonlinear emulators used MGCAMB.
                if mgpars["mu0"] > 2.0 * mgpars["sigma0"] + 1.0:
                    self.check_ranges = False
            else:
                raise ValueError("Mu-Sigma parameters are not properly specified.")

        # Instantiate MGrowth cosmology
        self.mg_cosmo = mg_models[self.gravity_model](background_mgrowth)

        # Assign self._compute_growth_generic once instead of an if-statement
        # later other mu_interpolators can be added (e.g., binned mu)
        if self.gravity_model == "musigma-de":
            self._compute_growth_generic = getattr(self, "_compute_growth_muinterp")
        elif self.gravity_model == "mu":
            self._compute_growth_generic = getattr(
                self, "_compute_growth_muinterp_scaledep"
            )
        else:
            self._compute_growth_generic = getattr(
                self, f"_compute_growth_{self.gravity_model}", None
            )
        if self._compute_growth_generic is None:
            raise ValueError(
                f"Growth computation for model '{self.gravity_model}' is not implemented."
            )

        # Compute MG and w0waCDM growth and assign linear growth parameters
        self._compute_growth(background_mgrowth)

    def _compute_mu_de_interp(self, mu0, omega0, w0, wa):
        omegaL = (
            (1.0 - omega0)
            * self.a_interp ** (-3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * (1.0 - self.a_interp))
        )
        omegaL0 = 1.0 - omega0
        E2 = omega0 / self.a_interp**3 + omegaL
        mu_de = 1.0 + mu0 * (omegaL / E2) / omegaL0
        mu_interpolator = interpolate.interp1d(
            self.a_interp,
            mu_de,
            bounds_error=False,
            kind="cubic",
            fill_value=(mu_de[0], mu_de[-1]),
        )
        # mu as a function of a (scale-factor)
        return mu_interpolator

    def _compute_mu_de_scaledep_interp(self, mu0, c1, lam, omega0, w0, wa):
        k_arr = self.k * self.background.h
        omegaL = (
            (1.0 - omega0)
            * self.a_interp ** (-3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * (1.0 - self.a_interp))
        )
        omegaL0 = 1.0 - omega0
        E2 = omega0 / self.a_interp**3 + omegaL
        E = np.sqrt(E2)
        # Note that we use the same convention as in ReACTEmu_cosmology, to convert to DESI parametrisation
        # lam = lam_DESI/2997.92458
        mu_de_k = np.array(
            [
                1.0
                + mu0
                * (omegaL / E2)
                / omegaL0
                * (1.0 + c1 * (lam * E / k_i) ** 2)
                / (1.0 + (lam * E / k_i) ** 2)
                for k_i in k_arr
            ]
        )
        mu_interpolator = [
            interpolate.interp1d(
                self.a_interp,
                row,
                bounds_error=False,
                kind="cubic",
                fill_value=(row[0], row[-1]),
            )
            for row in mu_de_k
        ]
        # mu as an array of interpolators in a (scale-factor) for fixed values of k
        return mu_interpolator

    def _compute_sigma_de_interp(self, sigma0, omega0, w0, wa):
        omegaL = (
            (1.0 - omega0)
            * (1.0 + self.z_sorted) ** (3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * self.z_sorted / (1.0 + self.z_sorted))
        )
        omegaL0 = 1.0 - omega0
        E2 = omega0 * (1.0 + self.z_sorted) ** 3 + omegaL
        sigma_de = 1.0 + sigma0 * (omegaL / E2) / omegaL0
        # sigma_interpolator = interpolate.interp1d(self.z_sorted, sigma_de, bounds_error=False,
        #        kind='cubic',
        #        fill_value=(sigma_de[0], sigma_de[-1]))
        sigma_de = np.repeat(sigma_de[:, None], self.k_len, axis=1)
        sigma_interpolator = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, sigma_de, kx=1, ky=1
        )
        # sigma as a function of z (redshift) and k
        return sigma_interpolator

    def _compute_sigma_de_scaledep_interp(self, sigma0, c2, lam, omega0, w0, wa):
        omegaL = (
            (1.0 - omega0)
            * (1.0 + self.z_sorted) ** (3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * self.z_sorted / (1.0 + self.z_sorted))
        )
        omegaL0 = 1.0 - omega0
        E2 = omega0 * (1.0 + self.z_sorted) ** 3 + omegaL
        E = np.sqrt(E2)
        k_arr = self.k * self.background.h
        z_ax = slice(None), None  # (n_z, 1)
        k_ax = None, slice(None)  # (1,  n_k)
        # Note that we use the same convention as in ReACTEmu_cosmology, to convert to DESI parametrisation
        # lam = lam_DESI/2997.92458
        sigma_de = 1.0 + sigma0 * (omegaL[z_ax] / E2[z_ax]) / omegaL0 * (
            1.0 + c2 * (lam * E[z_ax] / k_arr[k_ax]) ** 2
        ) / (1.0 + (lam * E[z_ax] / k_arr[k_ax]) ** 2)
        # sigma_de shape: (n_z, n_k)
        sigma_interpolator = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, sigma_de, kx=1, ky=1
        )
        return sigma_interpolator

    def _compute_growth_w0wacdm(self):
        D_raw, f_raw = self.mg_cosmo.growth_parameters()
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_dgp(self):
        D_raw, f_raw = self.mg_cosmo.growth_parameters(omegarc=self.mgpars["omega_rc"])
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_fr(self):
        # Full k-dependent growth for f(R)
        # MGrowth takes k in h/Mpc
        D_raw, f_raw = self.mg_cosmo.growth_parameters(
            k_arr=self.k * self.background.h, fR0=self.mgpars["fr0"]
        )
        # Transpose to get the arrays in (z,k), i.e. same shape as input power spectrum
        D_raw, f_raw = D_raw.T, f_raw.T
        return D_raw[::-1, :], f_raw[::-1, :]

    def _compute_growth_ds(self):
        # Correction due to non-universality of Dark Scattering,
        # i.e. dark energy interacts with dark matter only, not with baryons and neutrinos.
        h = self.background.H0 / 100
        omega_cdm = self.background.Omega_cdm0
        omega0 = (
            omega_cdm + self.background.Omega_b0 + self.background.mnu / 93.14 / h**2
        )
        Rc = omega_cdm / omega0
        unit_conv = 0.0194407
        xi = self.mgpars["xi"]
        w0 = self.background.w0
        # approximation from Carrilho et al. 2021 https://arxiv.org/pdf/2111.13598
        xi_corrected = (
            xi
            * Rc
            / (1.0 + unit_conv * h * (1.0 - omega0) * (1.0 + w0) * xi * (1.0 - Rc))
        )

        D_raw, f_raw = self.mg_cosmo.growth_parameters(xi=xi_corrected)
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_gamma(self):
        D_raw, f_raw = self.mg_cosmo.growth_parameters(gamma=self.mgpars["gamma0"])
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_gammaz(self):
        D_raw, f_raw = self.mg_cosmo.growth_parameters(
            gamma0=self.mgpars["gamma0"], gamma1=self.mgpars["gamma1"]
        )
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_muinterp(self):
        D_raw, f_raw = self.mg_cosmo.growth_parameters(mu_interp=self.mu_interp)
        # order in z instead of a and add a dimension for k
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(
            f_raw[::-1, None], self.k_len, axis=1
        )

    def _compute_growth_muinterp_scaledep(self):
        d_f_i = [
            self.mg_cosmo.growth_parameters(mu_interp=mu_interpolator_k_i)
            for mu_interpolator_k_i in self.mu_interp
        ]
        da = np.array([d_i for d_i, _ in d_f_i])
        fa = np.array([f_i for _, f_i in d_f_i])
        # transpose to arrays in z and k
        dz = da[:, ::-1].T
        fz = fa[:, ::-1].T
        return dz, fz

    def _compute_growth(self, bg_dict):
        """Handle model-specific MGrowth and w0waCDM growth evaluation."""
        D_mg, f_mg = self._compute_growth_generic()

        # Get w0waCDM growth and construct array over k to be applied as normalisation
        base_cosmo = mgrowth.w0waCDM(bg_dict)
        D_base_raw, _ = base_cosmo.growth_parameters()
        D_base = D_base_raw[::-1]

        # Interpolate D(z, k) and f(z, k)
        self.dz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg, kx=1, ky=1
        )
        self.dz_norm_dz0_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_mg[0, :], kx=1, ky=1
        )
        self.fz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, f_mg, kx=1, ky=1
        )
        # normalised to LCDM or w0waCDM at redshift 0
        self.dz_norm_w0wacdm_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_base[0, None], kx=1, ky=1
        )

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Calculate the growth factor D(z, k) normalized to D(0).

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns
        -------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        return self.dz_norm_dz0_interp(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """Calculate the growth rate f(z, k) for given redshifts and wavenumbers.

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns
        -------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        return self.fz_interp(zs, ks)

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Compute the linear matter power spectrum.

        Parameters
        ----------
        ks : numpy.ndarray
            Wavenumber in h Mpc^{-1}.
        zs : numpy.ndarray
            Redshifts.

        Returns
        -------
        np.ndarray
            Linear matter power spectrum at the specified redshifts and scales.
        """
        ps_base = self.base.matter_power_spectrum(0.0, ks)
        return self.dz_norm_w0wacdm_interp(zs, ks) ** 2 * ps_base

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""
        if self.gravity_model == "fr" or self.gravity_model == "mu":
            ks = np.logspace(-3, 2, 512)
            pk_lin_z0 = self.matter_power_spectrum(0.0, ks)

            def bes_j_1(x):
                return np.sin(x) / x - np.cos(x)

            return np.sqrt(
                integrate.simpson(
                    (3 * bes_j_1(ks * 8) / (ks * 8) ** 2) ** 2
                    * ks**2
                    * pk_lin_z0
                    / 2
                    / np.pi**2,
                    ks,
                )
            )
        else:
            return self.dz_norm_w0wacdm_interp(0.0, 0.01)[0, 0] * self.base.sigma8_0()
