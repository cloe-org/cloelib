import numpy as np
from scipy.stats import skewnorm
from scipy.integrate import simpson as simps
from astropy import constants as ap_constants
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.special import j0, j1

from ...auxiliary import units
from .halo_statistics import HaloStatistics


class Profile:
    def __init__(
        self,
        halo_statistics: HaloStatistics,
        two_halo="None",
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        trunc_fact=3.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
    ):
        self.halo_statistics = halo_statistics
        self.two_halo = two_halo
        if self.two_halo not in ["None", "sum", "max"]:
            raise ValueError("Invalid 'two_halo' definition, %s." % self.two_halo)

        # offcentering
        self.offcentering = offcentering
        self.rms_off = rms_off
        self.f_off = f_off
        self.trunc_fact = trunc_fact

        # ???
        self.zs_max = zs_max
        self.mean_nz = mean_nz
        self.sigma_nz = sigma_nz
        self.alpha_nz = alpha_nz

        # true redshift array (integration variable)
        z_min = 1e-5
        z_max = (
            self.zs_max - 1e-5
        )  # correction needed for avoiding zero values in n_zs_norM computation
        self.z_div = 50
        self.zed = np.linspace(z_min, z_max, self.z_div + 1)

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.zed)
        self.nzs = self.n_zs(self.zed)
        self.r_interp = np.logspace(-10, 2.5, 200)

    @property
    def perturbations(self):
        r"""
        Returns the Perturbations class instance
        """
        return self.halo_statistics.perturbations

    @property
    def background(self):
        r"""
        Returns the Background class instance
        """
        return self.perturbations.background

    def sigma_crit(self, z, z_sources):
        r"""
        Critical surface mass density.

        Computes the critical surface mass density at the given
        lens and source redshifts.

        Parameters
        ----------
        z: float
            Lens redshift.
        z_sources: np.ndarray
            Source redshift points.

        Returns
        -------
        sigma_crit : float
            Critical surface mass density (unit: h * Msun / pc^2)
        """
        fact = (units.SPEED_OF_LIGHT / 1.0e3 / units.MPC_TO_KM) ** 2.0 / (
            4.0 * np.pi * units.GRAVITATIONAL_CONSTANT
        )  # Msun/Mpc
        d_a_sources = self.background.angular_diameter_distance(z_sources)  # Mpc
        d_a_l = self.background.angular_diameter_distance(z)  # Mpc
        d_m_l = (1.0 + z) * d_a_l
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_h = units.SPEED_OF_LIGHT / 1e3 / self.background.H0  # Mpc
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(
                    1.0
                    + self.background.Omega_k0
                    * (d_m_l[:, np.newaxis] ** 2.0 / d_h**2.0)
                )
                - d_m_l[:, np.newaxis]
                * np.sqrt(
                    1.0 + self.background.Omega_k0 * (d_m_sources**2.0 / d_h**2.0)
                )
            )
        )
        sig_crit = fact * (d_a_sources / (d_a_l[:, np.newaxis] * d_a_lens_source))

        return 1e-12 * sig_crit / self.background.h  # Msun pc^{-2} h

    def n_zs_norM(self, z):
        r"""
        Galaxy number density normalization.

        Computes the galaxy number density normalization given a lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.

        Returns
        -------
        n_zs_norM: float or np.ndarray
            Galaxy number density normalization per redshift
        """
        n_zs_norM = 1.0 / (
            skewnorm.cdf(self.zs_max, self.alpha_nz, self.mean_nz, self.sigma_nz)
            - skewnorm.cdf(z, self.alpha_nz, self.mean_nz, self.sigma_nz)
        )

        return n_zs_norM

    def n_zs(self, z):
        r"""
        Galaxy number density.

        Computes the galaxy number density given a lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.

        Returns
        -------
        n_zs: float or np.ndarray
            Galaxy number density per redshift
        """
        n_zs = np.zeros((len(z), self.z_div + 1))
        for z_ind, zed in enumerate(z):
            z_s = np.linspace(zed + 1.0e-5, self.zs_max, self.z_div + 1)
            n_zs[z_ind] = skewnorm.pdf(z_s, self.alpha_nz, self.mean_nz, self.sigma_nz)

        return n_zs

    def m_sig_crit_m1(self, z, zbin):
        r"""
        Effective inverse critical surface mass density.

        Computes the effective critical surface mass density at
        the given lens redshift.

        Parameters
        ----------
        z: float or np.ndarray
            Lens redshift.
        zbin: int
            Index of the lens redshift bin.

        Returns
        -------
        m_sigma_crit_m1: float
            Effective inverse critical surface mass density (units : pc^2 / Msun / h)
        """
        z_s = np.linspace(z + 1.0e-5, self.zs_max, self.z_div + 1, axis=1)
        sig_crit_m1[:] = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s[:])

        return self.nzsnorM[zbin] * simps(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def _surface_mass_density_cen(self, R, z, c, M, force_no_2h=False):
        r"""
        Centered surface mass density profile.

        Computes the centered surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        c: float
            Concentration.
        M: np.ndarray
            Mass (Msun / h).
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term

        Returns
        -------
        Sigma: np.ndarray
            Centered surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Delta_crit = np.atleast_1d(self.halo_statistics.get_Delta_crit(z))[
            :, np.newaxis
        ]
        rho_c = self.background.rho_crit(z[:, np.newaxis]) / self.background.h**2.0
        densityThreshold = Delta_crit * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)

        Sigma = self._surface_mass_density_profile(
            R[np.newaxis, np.newaxis, :],
            RDelta[:, :, np.newaxis],
            c,
            densityThreshold[:, :, np.newaxis],
        )

        if force_no_2h == False and self.two_halo == "sum":
            Sigma += self.surface_mass_density_2h(R, z, M)
        elif force_no_2h == False and self.two_halo == "max":
            Sigma_2h = self.surface_mass_density_2h(R, z, M)
            Sigma = np.maximum(Sigma, Sigma_2h)

        expected_shape = (
            len(np.atleast_1d(z.squeeze())),
            len(np.atleast_1d(M.squeeze())),
            len(np.atleast_1d(R.squeeze())),
        )
        assert (
            Sigma.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {Sigma.shape}"

        return Sigma

    def _surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        Centered one-halo surface mass density profile.

        Computes the centered one-halo surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma: np.ndarray
            Centered one-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        return NotImplementedError

    def surface_mass_density(self, R, z, c, M, force_no_2h=False, force_no_off=False):
        r"""
        Total surface mass density profile.

        Computes the total surface mass density profile at radius R,
        including the contribution from 2-halo term and miscetering.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        c: float
            Concentration.
        M: np.ndarray
            Mass (Msun / h).
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term
        force_no_off: bool
            if True, force the non-inclusion of the off-centering

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        if force_no_off == False and self.offcentering and self.rms_off >= 1.0e-4:
            Sigma_off = np.zeros_like(R)

            ir.Sigma_off(
                R,
                self.r_interp,
                self._surface_mass_density_cen(
                    self.r_interp, z, c, M, force_no_2h=False
                ),
                self.rms_off,
                Sigma_off,
            )

            Sigma_cen = self._surface_mass_density_cen(R, z, c, M, force_no_2h=False)
            return (1.0 - self.f_off) * Sigma_cen + self.f_off * Sigma_off

        else:
            return self._surface_mass_density_cen(R, z, c, M, force_no_2h)

    def excess_surface_mass_density(self, R, z, c, M):
        r"""
        Total excess surface mass density profile.

        Computes the total excess surface mass density profile at radius R,
        including the contribution from 2-halo term and miscetering.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        c: float
            Concentration.
        M: np.ndarray
            Mass (Msun / h).
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term
        force_no_off: bool
            if True, force the non-inclusion of the off-centering

        Returns
        -------
        DeltaSigma: np.ndarray
            Excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Delta_crit = np.atleast_1d(self.halo_statistics.get_Delta_crit(z))[
            :, np.newaxis
        ]
        rho_c = self.background.rho_crit(z[:, np.newaxis]) / self.background.h**2.0
        densityThreshold = Delta_crit * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)

        Sigma_mean = self._mean_surface_mass_density_profile(
            R[np.newaxis, np.newaxis, :],
            RDelta[:, :, np.newaxis],
            c,
            densityThreshold[:, :, np.newaxis],
        )
        Sigma = self._surface_mass_density_cen(R, z, c, M, force_no_2h=True)
        DeltaSigma = Sigma_mean - Sigma

        if self.two_halo == "sum":
            DeltaSigma += self.excess_surface_mass_density_2h(R, z, M)
        elif self.two_halo == "max":
            DeltaSigma_2h = self.excess_surface_mass_density_2h(R, z, M)
            DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        if self.offcentering:
            if self.rms_off >= 1.0e-4 and self.f_off >= 1.0e-4:
                R = np.asarray(R)
                DeltaSigma_off = np.zeros_like(R)

                ir.DeltaSigma_off(
                    R,
                    self.r_interp,
                    self.r_interp,
                    self._surface_mass_density_cen(
                        self.r_interp, z, c, M, force_no_2h=False
                    ),
                    self.rms_off,
                    DeltaSigma_off,
                )

                DeltaSigma *= 1.0 - self.f_off
                DeltaSigma += self.f_off * DeltaSigma_off

            elif self.rms_off < 1.0e-4 and self.f_off >= 1.0:
                DeltaSigma = np.zeros(len(DeltaSigma))

        return DeltaSigma

    def _mean_surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        Centered one-halo mean surface mass density profile.

        Computes the centered one-halo mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma_mean: np.ndarray
            Centered one-halo mean surface mass density (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        return NotImplementedError

    def _mass_density_2h(self, R, z, M, bias_z, bessel_term):
        r"""
        Surface or excess surface 2-halo density profile.

        Computes either the cosmological surface or excess surface
        2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        bias_z: np.ndarray
            Halo bias. If None, it is computed internally.
        bessel_term: function
            Bessel term of the integrand with the power spectrum.
            Used to return the surface density or the excess surface density.

        Returns
        -------
        profile: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        # Ensure proper array shapes (z, M, R)
        z = np.asarray(z)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
        M = np.asarray(M)[np.newaxis, :, np.newaxis]  # shape (1, nM, 1)
        R = np.asarray(R)[np.newaxis, np.newaxis, :]  # shape (1, 1, nR)

        # Squeeze z and M
        z_squeeze = np.atleast_1d(z.squeeze())
        M_squeeze = np.atleast_1d(M.squeeze())

        # Calculate base quantities
        D_A = self.background.angular_diameter_distance(z_squeeze)[
            :, np.newaxis, np.newaxis
        ]

        theta = R / D_A

        rho_m = (
            self.background.Omega_m(z_squeeze, nonu=False)
            * self.background.rho_crit(z_squeeze)
            / self.background.h**2
        )[:, np.newaxis, np.newaxis]

        # Power spectrum interpolation
        kl_min, kl_max = 1e-4, 1e2
        kl_array = np.logspace(np.log10(kl_min), np.log10(kl_max), 500)

        if len(z_squeeze) < 10:
            z_for_interp = np.linspace(z.min() * 0.9, z.max() * 1.1, 10)
        else:
            z_for_interp = z_squeeze

        Pk_interp = interpolate.RectBivariateSpline(
            z_for_interp,
            kl_array,
            self.perturbations.matter_power_spectrum(
                z_for_interp[:, np.newaxis],
                kl_array,
                hubble_units=True,
                k_hunit=True,
                nonu=self.halo_statistics.nonu,
            ),
        )

        # Ensure bias has shape (nz, nM, 1)
        if bias_z is None:
            bias_z = self.halo_statistics.bias(z_squeeze, M_squeeze)[:, :, np.newaxis]
        else:
            bias_z = np.asarray(bias_z)[:, :, np.newaxis]

        # Integrand function
        def integrand(kl):
            kl = np.atleast_1d(kl)
            ll = kl[:, np.newaxis, np.newaxis, np.newaxis] * (1.0 + z) * D_A
            Pk_vals = Pk_interp(z_squeeze, kl).T
            return (
                bessel_term(ll, theta)
                * Pk_vals[:, :, np.newaxis, np.newaxis]
                * (1.0 + z)
                * D_A
            )

        # Integration
        profile = quad_vec(integrand, kl_min, kl_max, epsrel=1e-1)[0]
        profile = np.squeeze(profile, axis=0)

        # Final strictly 3D calculation
        denominator = 2.0 * np.pi * (1.0 + z) ** 3.0 * D_A**2.0
        profile = (1.0e-12 * rho_m * bias_z * profile) / denominator

        expected_shape = (
            len(np.atleast_1d(z.squeeze())),
            len(np.atleast_1d(M.squeeze())),
            len(np.atleast_1d(R.squeeze())),
        )
        assert (
            profile.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {profile.shape}"

        return profile

    def surface_mass_density_2h(self, R, z, M, bias_z=None):
        r"""
        Surface 2-halo density profile.

        Computes the cosmological surface 2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        bias_z: np.ndarray (optional)
            Halo bias. If None, it is computed internally.

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        return self._mass_density_2h(R, z, M, bias_z, bessel_term=self._bessel_term)

    def _bessel_term(self, ll, theta):
        return j0(ll * theta) * ll

    def _bessel_term_exess(self, ll, theta):
        j2 = 2.0 / (ll * theta) * j1(ll * theta) - j0(ll * theta)
        return j2 * ll

    def excess_surface_mass_density_2h(self, R, z, M, bias_z=None):
        r"""
        Excess surface 2-halo density profile.

        Computes the cosmological excess surface 2-halo
        density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        bias_z: np.ndarray (optional)
            Halo bias. If None, it is computed internally.

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        return self._mass_density_2h(
            R, z, M, bias_z, bessel_term=self._bessel_term_exess
        )

    def _f_term(self, x):
        r"""
        One-halo centered profile F term.

        Computes the one-halo centered profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile F term.
        """
        return NotImplementedError

    def _g_term(self, x):
        r"""
        One-halo centered profile G term.

        Computes the one-halo centered profile G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        float
            One-Halo profile G term.
        """
        return NotImplementedError


class ProfileNFW(Profile):
    def _f_term(self, x):
        r"""
        NFW profile F term.

        Computes the NFW profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
            One-Halo NFW F term.

        Notes
        -----
        Implementation of second part of Eq. 4 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return (1.0 - np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)) / (x**2.0 - 1.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)) / (x**2.0 - 1.0)

    def _g_term(self, x):
        r"""
        NFW profile G term.

        Computes the NFW profile G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        G_NFW: float
            One-Halo NFW G term.

        Notes
        -----
        Implementation of Eq. 5 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0 + np.log(1.0 / 2.0)
        if x > 1.0:
            return np.log(x / 2.0) + np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def _surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        NFW surface mass density profile.

        Computes the NFW surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma: np.ndarray
            NFW surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Rs = RDelta / c
        x = R / Rs

        F = np.vectorize(self._f_term)(x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta * c**3.0 / (3.0 * m_nfw)

        Sigma = 2.0 * rho_s * Rs * F * 1.0e-12

        return Sigma

    def _mean_surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        NFW mean surface mass density profile.

        Computes the NFW mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma_mean: np.ndarray
            NFW mean surface mass density (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Rs = RDelta / c
        x = R / Rs

        G = np.vectorize(self._g_term)(x)

        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta * c**3.0 / (3.0 * m_nfw)

        return 4.0 * rho_s * Rs * (G / x**2.0) * 1.0e-12


class ProfileBMO(Profile):
    def _f_term(self, x):
        r"""
        BMO profile F term.

        Computes the BMO profile F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_BMO : float
            One-Halo BMO F term.

        Notes
        -----
        Implementation of Eq. A.5 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0
        if x > 1.0:
            return np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def _g_term(self, x):
        r"""
        BMO profile G term.

        Computes the BMO profile G term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        G_BMO : float
            One-Halo BMO G term.

        Notes
        -----
        Implementation of Eq. A.28 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return (self._f_term(x) - 1.0) / (1.0 - x**2.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - self._f_term(x)) / (x**2.0 - 1.0)

    def _surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        BMO surface mass density profile.

        Computes the BMO surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma: np.ndarray
            BMO surface mass density profile (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta * c**3.0 / (3.0 * m_bmo)

        const = rho_s_bmo * Rs

        G = np.vectorize(self._g_term)(x)
        F = np.vectorize(self._f_term)(x)

        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0
        term2 = 2.0 * (tau**2.0 + 1.0) * G
        term3 = 8.0 * F
        term4 = (tau**4.0 - 1.0) / (tau**2.0 * (tau**2.0 + x**2.0))
        term5 = (
            np.pi
            * (4.0 * (tau**2.0 + x**2.0) + tau**2.0 + 1.0)
            / (tau**2.0 + x**2.0) ** (3.0 / 2.0)
        )
        term6 = (
            tau**2.0 * (tau**4.0 - 1.0)
            + (tau**2.0 + x**2.0) * (3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0)
        ) / (tau**3.0 * (tau**2.0 + x**2.0) ** (3.0 / 2.0))

        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        Sigma = 1e-12 * const * term1 * (term2 + term3 + term4 - term5 + term6 * L)
        return Sigma

    def _mean_surface_mass_density_profile(self, R, RDelta, c, Delta):
        r"""
        BMO mean surface mass density profile.

        Computes the BMO mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        c: float
            Concentration.
        Delta: np.ndarray
            Critical overdensity.

        Returns
        -------
        Sigma_mean: np.ndarray
            BMO mean surface mass density (units : h * Msun / pc**2).
            Shape: (len(z), len(M), len(R)).
        """
        Rs = RDelta / c
        x = R / Rs

        Rt = self.trunc_fact * RDelta
        tau = Rt / Rs

        m_bmo = (
            tau**2.0
            / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
            * (
                c
                * (tau**2.0 + 1.0)
                * (
                    c * (c + 1.0)
                    - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                    - 2.0 * tau**4.0
                )
                + tau
                * (c + 1.0)
                * (tau**2.0 + c**2.0)
                * (
                    2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                    + tau
                    * (tau**2.0 - 3.0)
                    * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                )
            )
        )

        rho_s_bmo = Delta * c**3.0 / (3.0 * m_bmo)

        const = 2.0 * np.pi * rho_s_bmo * Rs**3.0
        term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0

        F = np.vectorize(self._f_term)(x)
        term2 = 2.0 * (tau**2.0 + 1.0 + 4.0 * (x**2.0 - 1.0)) * F

        G = np.vectorize(self._g_term)(x)
        term3 = (
            np.pi * (3.0 * tau**2.0 - 1.0) + 2.0 * tau * (tau**2.0 - 3.0) * np.log(tau)
        ) / tau

        term4 = tau**3.0 * np.sqrt(tau**2.0 + x**2.0)
        term5 = -(tau**3.0) * np.pi * (4.0 * (tau**2.0 + x**2.0) - tau**2.0 - 1.0)
        term6 = -(tau**2.0) * (tau**4.0 - 1.0) + +(tau**2.0 + x**2.0) * (
            3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0
        )
        L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

        M_proj = const * term1 * (term2 + term3 + (term5 + term6 * L) / term4)

        return M_proj / (np.pi * R**2.0) * 1.0e-12
