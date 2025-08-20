import numpy as np
from scipy.stats import skewnorm
from scipy.integrate import simpson as simps
from astropy import units as ap_units
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.special import j0, j1
from scipy import interpolate

from ...auxiliary import units
from .halo_statistics import HaloStatistics
from cloelib.cosmology import derived_cosmology as dc


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)


class Profile:
    def __init__(
        self,
        halostatistics: HaloStatistics,
        k: np.ndarray = np.geomspace(1e-4, 10, 500),
        z: np.ndarray = np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        r_interp: np.ndarray = np.logspace(-10, 2.5, 200),
        two_halo: str = "None",
        offcentering: bool = False,
        rms_off: float = 0.0,
        f_off: float = 0.0,
        trunc_fact: float = 3.0,
        zs_max: float = 2.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
        use_interpolation: bool = True,
    ):
        self.halostatistics = halostatistics
        self.k = k
        self.z = z
        self.r_interp = r_interp

        self._validate_two_halo(two_halo)
        self.two_halo = two_halo

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

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.z)
        self.nzs = self.n_zs(self.z)

        # set interpolation usage
        self.interp_angular_dist = None
        if use_interpolation:
            self.interpolate_angular_diameter_distance()
        self.use_interpolation = use_interpolation

    def _validate_two_halo(self, two_halo):
        if two_halo not in ("None", "sum", "max"):
            raise ValueError("Invalid 'two_halo' definition, %s." % two_halo)

    @property
    def perturbations(self):
        r"""
        Returns the Perturbations class instance
        """
        return self.halostatistics.perturbations

    @property
    def background(self):
        r"""
        Returns the Background class instance
        """
        return self.perturbations.background

    @property
    def use_interpolation(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        return self.__use_interpolation

    @use_interpolation.setter
    def use_interpolation(self, use_interpolation):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        if use_interpolation:
            self.angular_diameter_distance = self.interp_angular_dist
        else:
            self.angular_diameter_distance = self.background.angular_diameter_distance
        self.__use_interpolation = use_interpolation

    def interpolate_angular_diameter_distance(self):
        r"""Create internal interpolation of angular diameter distance."""
        self.interp_angular_dist = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(self.z.min(), self.z.max()+1.e-5, len(self.z)), y=self.background.angular_diameter_distance(np.linspace(self.z.min(),
            self.z.max()+1.e-5, len(self.z))), ext=2)

    def convert_distance(
        self, distance, units_in, units_out, angular_diameter_distance=None
    ):
        r"""Convert distances

        Parameters
        ----------
        distance: np.ndarray
            Input projected distances
        units_in: str
            Unit for the input projected distance. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        units_out: str
            Unit for the output projected distance. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        angular_diameter_distance: float, np.ndarray
            Angular diameter distance (units: Mpc/h) to be used for converting
            between angular and physical units. If array, it
            should be in the shape (z.size, 1).

        Returns
        -------
        np.ndarray
            Distance in output units. If z is array and physical to
            angular conversion used, output shape is (z.size, distance.size).
        """
        angular_units_dict = {
            "radians": ap_units.rad,
            "degrees": ap_units.deg,
            "arcmin": ap_units.arcmin,
            "arcsec": ap_units.arcsec,
        }
        _valid_units = ["mpc/h", *angular_units_dict.keys()]
        if units_in.lower() not in _valid_units:
            raise ValueError(f"units_in (={units_in}) must be in {_valid_units}")
        if units_out.lower() not in _valid_units:
            raise ValueError(f"units_out (={units_out}) must be in {_valid_units}")

        if units_in.lower() == units_out.lower():
            return distance

        if units_out.lower() not in angular_units_dict:
            # converting to mpc/h
            theta = (
                (distance * angular_units_dict[units_in]).to(ap_units.rad).value
            )  # distance in radians
            out = theta * angular_diameter_distance
        elif units_in.lower() not in angular_units_dict:
            # converting to angular units
            theta = distance / angular_diameter_distance  # distance in radians
            out = (theta * ap_units.rad).to(angular_units_dict[units_out]).value
        else:
            out = (
                (distance * angular_units_dict[units_in])
                .to(angular_units_dict[units_out])
                .value
            )

        return out

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
        d_a_sources = self.angular_diameter_distance(z_sources)  # Mpc
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_a_lens = self.angular_diameter_distance(z)[:, np.newaxis]  # Mpc
        d_m_lens = (1.0 + z[:, np.newaxis]) * d_a_lens
        d_h = units.SPEED_OF_LIGHT / 1e3 / self.background.H0  # Mpc
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_lens / d_h) ** 2.0)
                - d_m_lens
                * np.sqrt(1.0 + self.background.Omega_k0 * (d_m_sources / d_h) ** 2.0)
            )
        )
        sig_crit = fact * d_a_sources / (d_a_lens * d_a_lens_source)

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
        n_zs = np.zeros((z.size, len(self.z)))
        for z_ind, _z in enumerate(z):
            z_s = np.linspace(_z + 1.0e-5, self.zs_max, len(self.z))
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
        z_s = np.linspace(z + 1.0e-5, self.zs_max, len(self.z), axis=1)
        sig_crit_m1 = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s)

        return self.nzsnorM[zbin] * simps(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def surface_mass_density(
        self,
        R,
        z,
        M,
        c,
        bias_z=None,
        radius_units="Mpc/h",
    ):
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
        M: np.ndarray
            Mass (Msun / h).
        c: float
            Concentration.
        bias_z: np.ndarray
            Halo bias used for the 2h term. If None, it is computed internally,
            else has to be shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if self.offcentering and self.rms_off >= 1.0e-4:
            Sigma_off = np.zeros_like(R)

            ir.Sigma_off(
                R,
                self.r_interp,
                self._surface_mass_density_cen(
                    self.r_interp, z, M, c, self.two_halo, radius_units=radius_units
                ),
                self.rms_off,
                Sigma_off,
            )

            Sigma_cen = self._surface_mass_density_cen(
                R, z, M, c, self.two_halo, radius_units=radius_units
            )
            return (1.0 - self.f_off) * Sigma_cen + self.f_off * Sigma_off
        else:
            return self._surface_mass_density_cen(
                R, z, M, c, self.two_halo, bias_z, radius_units=radius_units
            )

    def excess_surface_mass_density(
        self, R, z, M, c, bias_z=None, radius_units="Mpc/h"
    ):
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
        bias_z: np.ndarray
            Halo bias used for the 2h term. If None, it is computed internally,
            else has to be shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        # centered 1h term
        Sigma_mean = self._model_mean_surface_mass_density_profile(
            *self._surface_mass_density_args(R, z, M, radius_units=radius_units), c
        )
        Sigma = self._surface_mass_density_cen(
            R, z, M, c, two_halo="None", radius_units=radius_units
        )
        DeltaSigma = Sigma_mean - Sigma

        # centered 2h term
        if self.two_halo != "None":
            DeltaSigma_2h = self._excess_surface_mass_density_2h(
                R, z, M, bias_z, radius_units
            )
            if self.two_halo == "sum":
                DeltaSigma += DeltaSigma_2h
            elif self.two_halo == "max":
                DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        # offcentered terms
        if self.offcentering and self.rms_off >= 1.0e-4 and self.f_off >= 1.0e-4:
            # non negligible offcentering
            R = np.asarray(R)
            DeltaSigma_off = np.zeros_like(R)

            ir.DeltaSigma_off(
                R,
                self.r_interp,
                self.r_interp,
                self._surface_mass_density_cen(
                    self.r_interp,
                    z,
                    c,
                    M,
                    two_halo=self.two_halo,
                    radius_units=radius_units,
                ),
                self.rms_off,
                DeltaSigma_off,
            )

            DeltaSigma = (1.0 - self.f_off) * DeltaSigma + self.f_off * DeltaSigma_off

        elif self.offcentering and self.rms_off < 1.0e-4 and self.f_off >= 1.0:
            # extreme offcentering
            DeltaSigma = np.zeros(len(DeltaSigma))

        return DeltaSigma

    def _surface_mass_density_args(self, R, z, M, radius_units="Mpc/h"):
        r"""
        Prepare arguments for _model_surface_mass_density_profile and
        _model_mean_surface_mass_density_profile with correct shapes.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        R_outshape: np.ndarray
            Radius (units: Mpc / h) with shape (1, 1, R.size) if radius_units="Mpc/h"
            else (z.size, 1, R.size)
        RDelta: np.ndarray
            Radius of overdensity (units: Mpc / h) with shape (z.size, M.size, 1)
        densityThreshold: np.ndarray
            Threshold density (units : h * Msun / Mpc**2)  with shape (z.size, 1, 1)
        """
        densityThreshold = np.atleast_1d(
            self.halostatistics.get_Delta_crit(z)
            * dc.rho_crit(self.background, z)
            / self.background.h**2.0
        )[:, np.newaxis, np.newaxis]

        RDelta = (
            3.0 * M[np.newaxis, :, np.newaxis] / 4.0 / np.pi / densityThreshold
        ) ** (1.0 / 3.0)

        if radius_units.lower() != "mpc/h":
            D_A = (
                self.background.angular_diameter_distance(z) * self.background.h
            )  # Mpc / h
            R_outshape = self.convert_distance(
                R, radius_units, "Mpc/h", D_A[:, np.newaxis]
            )[:, np.newaxis]
        else:
            R_outshape = R[np.newaxis, np.newaxis, :]

        return R_outshape, RDelta, densityThreshold

    def _check_profile_shape(self, z, M, R, profile):
        expected_shape = (
            np.atleast_1d(z).size,
            np.atleast_1d(M).size,
            np.atleast_1d(R).size,
        )
        assert (
            profile.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {profile.shape}"

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        Centered one-halo surface mass density profile.

        Computes the centered one-halo surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            Centered one-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        raise NotImplementedError

    def _surface_mass_density_cen(
        self, R, z, M, c, two_halo="auto", bias_z=None, radius_units="Mpc/h"
    ):
        r"""
        Centered surface mass density profile.

        Computes the centered surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        c: float
            Concentration.
        two_halo: str
            Application of the 2-halo term, options are "None", "sum", "max".
        bias_z: np.ndarray
            Halo bias used for the 2h term. If None, it is computed internally,
            else has to be shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Centered surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Sigma = self._model_surface_mass_density_profile(
            *self._surface_mass_density_args(R, z, M, radius_units=radius_units), c
        )

        self._validate_two_halo(two_halo)
        if two_halo != "None":
            Sigma_2h = self._surface_mass_density_2h(R, z, M, bias_z, radius_units)
            if two_halo == "sum":
                Sigma += Sigma_2h
            elif two_halo == "max":
                Sigma = np.maximum(Sigma, Sigma_2h)

        self._check_profile_shape(z, M, R, Sigma)

        return Sigma

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
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
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            Centered one-halo mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        raise NotImplementedError

    def _func_mass_density_2h(
        self, R, z, M, bias_z, bessel_function, radius_units="Mpc/h"
    ):
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
            Halo bias. If None, it is computed internally,
            else has to be shape (z.size, M.size).
        bessel_function: function
            Bessel function that goes in the integrand with the power spectrum.
            Used to return the surface density or the excess surface density.
            It should take (ll*theta) as input.

        Returns
        -------
        profile: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        # Calculate base quantities
        D_A = self.background.angular_diameter_distance(z) * self.background.h

        # Ensure proper array shapes (z, M, R)
        z_outshape = np.asarray(z)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
        D_A_outshape = D_A[:, np.newaxis, np.newaxis]
        rho_m_outshape = (
            self.background.Omega_m(z)
            * dc.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis, np.newaxis]

        # Ensure bias has shape (nz, nM, 1)
        if bias_z is None:
            bias_z = self.halostatistics.bias(z, M)
        bias_z_outshape = np.asarray(bias_z)[:, :, np.newaxis]

        # Two point correlation part

        ## 1. Power spectrum interpolation

        kl_array = self.k

        ## 2. Get radial distance in radians
        _theta = self.convert_distance(R, radius_units, "radians", D_A[:, np.newaxis])
        theta_outshape = _theta[:, np.newaxis]
        if radius_units.lower() != "mpc/h":
            # in this case, theta_outshape was missing z dimension
            theta_outshape = theta_outshape[np.newaxis, np.newaxis, :, 0]

        ## 3. Integrand function
        def integrand(kl):
            ll = kl * (1.0 + z_outshape) * D_A_outshape
            Pk_vals = self.halostatistics.matter_power_spectrum(z, kl)[:, np.newaxis]
            return bessel_function(ll * theta_outshape) * ll * Pk_vals

        ## 4. Integration
        two_point_corr_outshape = (
            quad_vec(integrand, kl_array.min(), kl_array.max(), epsrel=1e-1)[0]
            * (1.0 + z_outshape)
            * D_A_outshape
        )

        # Final strictly 3D calculation
        profile = (
            1.0e-12 * rho_m_outshape * bias_z_outshape * two_point_corr_outshape
        ) / (2.0 * np.pi * (1.0 + z_outshape) ** 3.0 * D_A_outshape**2.0)

        self._check_profile_shape(z, M, R, profile)

        return profile

    def _surface_mass_density_2h(self, R, z, M, bias_z=None, radius_units="Mpc/h"):
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
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._func_mass_density_2h(
            R, z, M, bias_z, bessel_function=j0, radius_units=radius_units
        )

    def _excess_surface_mass_density_2h(
        self, R, z, M, bias_z=None, radius_units="Mpc/h"
    ):
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
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._func_mass_density_2h(
            R,
            z,
            M,
            bias_z,
            bessel_function=_bessel_j2,
            radius_units=radius_units,
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
        raise NotImplementedError

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
        raise NotImplementedError


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

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        NFW surface mass density profile.

        Computes the NFW surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            NFW surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        Rs = RDelta / c
        x = R / Rs

        F = np.vectorize(self._f_term)(x)
        m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
        rho_s = Delta * c**3.0 / (3.0 * m_nfw)

        Sigma = 2.0 * rho_s * Rs * F * 1.0e-12

        return Sigma

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
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
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            NFW mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
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

    def _model_surface_mass_density_profile(self, R, RDelta, Delta, c):
        r"""
        BMO surface mass density profile.

        Computes the BMO surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            BMO surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
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

    def _model_mean_surface_mass_density_profile(self, R, RDelta, Delta, c):
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
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            BMO mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
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
