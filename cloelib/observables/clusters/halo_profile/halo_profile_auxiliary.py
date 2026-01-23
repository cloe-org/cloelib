import numpy as np
from astropy import units as ap_units
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.special import j0, j1
from scipy.stats import skewnorm

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.halo_model import HaloModel


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)


class HaloProfileAuxiliary:
    def __init__(
        self,
        halo_model: HaloModel,
        z=np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        zs_max: float = 2.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
    ):
        r"""Auxiliary class computing quantities used in mass profile models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        halo_model : HaloModel
            An object from the `HaloModel` class.
        """
        self.halo_model = halo_model

        # ???
        self.z = z
        self.zs_max = zs_max
        self.mean_nz = mean_nz
        self.sigma_nz = sigma_nz
        self.alpha_nz = alpha_nz

        # interpolate the angular diameter distance
        self.angular_diameter_distance = interpolate.InterpolatedUnivariateSpline(
            x=np.linspace(self.z.min(), self.z.max() + 1.0e-5, len(self.z)),
            y=self.background.angular_diameter_distance(
                np.linspace(self.z.min(), self.z.max() + 1.0e-5, len(self.z))
            ),
            ext=2,
        )

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.z)
        self.nzs = self.n_zs(self.z)

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.halo_model.background

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
            skewnorm.cdf(
                self.zs_max,
                self.alpha_nz,
                self.mean_nz,
                self.sigma_nz,
            )
            - skewnorm.cdf(
                z,
                self.alpha_nz,
                self.mean_nz,
                self.sigma_nz,
            )
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
            n_zs[z_ind] = skewnorm.pdf(
                z_s,
                self.alpha_nz,
                self.mean_nz,
                self.sigma_nz,
            )

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
            self.halo_model.get_Delta_crit(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2.0
        )[:, np.newaxis, np.newaxis]

        RDelta = (
            3.0 * M[np.newaxis, :, np.newaxis] / 4.0 / np.pi / densityThreshold
        ) ** (1.0 / 3.0)

        if radius_units.lower() != "mpc/h":
            D_A = self.angular_diameter_distance(z) * self.background.h  # Mpc / h
            R_outshape = self.convert_distance(
                R, radius_units, "Mpc/h", D_A[:, np.newaxis]
            )[:, np.newaxis]
        else:
            R_outshape = R[np.newaxis, np.newaxis, :]

        return R_outshape, RDelta, densityThreshold

    def _check_profile_shape(self, R, z, M, profile):
        """
        Check the shape of the mass profile.
        """
        expected_shape = (
            np.atleast_1d(z).size,
            np.atleast_1d(M).size,
            np.atleast_1d(R).size,
        )
        assert (
            profile.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {profile.shape}"

    def _generic_mass_density_2h(
        self, R, z, M, halo_bias, bessel_function, radius_units="Mpc/h"
    ):
        r"""
        Surface or excess surface 2-halo density profile.

        Computes either the cosmological surface or excess surface
        (depending on the input Bessel function) 2-halo density profile.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
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
        D_A = self.angular_diameter_distance(z) * self.background.h

        # Ensure proper array shapes (z, M, R)
        z_outshape = np.asarray(z)[:, np.newaxis, np.newaxis]  # shape (nz, 1, 1)
        D_A_outshape = D_A[:, np.newaxis, np.newaxis]
        rho_m_outshape = (
            self.background.Omega_m(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis, np.newaxis]

        # Ensure bias has shape (nz, nM, 1)
        if halo_bias is None:
            raise ValueError(
                "halo_bias must be provided explicitly when computing the 2-halo term."
            )

        halo_bias = np.asarray(halo_bias)
        nz = np.atleast_1d(z).size
        nM = np.atleast_1d(M).size
        if halo_bias.shape != (nz, nM):
            raise ValueError(
                f"halo_bias must have shape (len(z), len(M)) = ({nz}, {nM}), "
                f"got {halo_bias.shape}"
            )

        halo_bias_outshape = halo_bias[:, :, np.newaxis]

        # Two point correlation part

        ## 1. Power spectrum interpolation

        kl_array = self.halo_model.k

        ## 2. Get radial distance in radians
        _theta = self.convert_distance(R, radius_units, "radians", D_A[:, np.newaxis])
        theta_outshape = _theta[:, np.newaxis]
        if radius_units.lower() != "mpc/h":
            # in this case, theta_outshape was missing z dimension
            theta_outshape = theta_outshape[np.newaxis, np.newaxis, :, 0]

        ## 3. Integrand function
        def integrand(kl):
            ll = kl * (1.0 + z_outshape) * D_A_outshape
            Pk_vals = self.halo_model.matter_power_spectrum(z, kl)[:, np.newaxis]
            return bessel_function(ll * theta_outshape) * ll * Pk_vals

        ## 4. Integration
        two_point_corr_outshape = (
            quad_vec(integrand, kl_array.min(), kl_array.max(), epsrel=1e-1)[0]
            * (1.0 + z_outshape)
            * D_A_outshape
        )

        # Final strictly 3D calculation
        profile = (
            1.0e-12 * rho_m_outshape * halo_bias_outshape * two_point_corr_outshape
        ) / (2.0 * np.pi * (1.0 + z_outshape) ** 3.0 * D_A_outshape**2.0)

        return profile

    def surface_mass_density_2h(self, R, z, M, halo_bias, radius_units="Mpc/h"):
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
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, M, halo_bias, bessel_function=j0, radius_units=radius_units
        )

    def excess_surface_mass_density_2h(self, R, z, M, halo_bias, radius_units="Mpc/h"):
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
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, M, halo_bias, bessel_function=_bessel_j2, radius_units=radius_units
        )

    def include_surface_mass_density_2h(
        self, Sigma_1h, inclusion_type, R, z, M, halo_bias, radius_units="Mpc/h"
    ):
        r"""
        Include the contribution of the cosmological 2-halo term.

        Include the contribution of the 2-halo term to the surface density profile.

        Parameters
        ----------
        Sigma_1h : np.ndarray
            One-halo surface mass density.
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Total surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if inclusion_type not in ("sum", "max"):
            raise ValueError(
                "Invalid 'inclusion_type' definition, %s." % inclusion_type
            )

        Sigma_2h = self.surface_mass_density_2h(R, z, M, halo_bias, radius_units)

        if inclusion_type == "sum":
            return Sigma_1h + Sigma_2h
        elif inclusion_type == "max":
            return np.maximum(Sigma_1h, Sigma_2h)

    def include_excess_surface_mass_density_2h(
        self, DeltaSigma_1h, inclusion_type, R, z, M, halo_bias, radius_units="Mpc/h"
    ):
        r"""
        Include the contribution of the cosmological 2-halo term.

        Include the contribution of the 2-halo term to the excess surface density profile.

        Parameters
        ----------
        DeltaSigma_1h : np.ndarray
            One-halo surface mass density.
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Total excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        if inclusion_type not in ("sum", "max"):
            raise ValueError(
                "Invalid 'inclusion_type' definition, %s." % inclusion_type
            )

        DeltaSigma_2h = self.excess_surface_mass_density_2h(
            R, z, M, halo_bias, radius_units
        )

        if inclusion_type == "sum":
            return DeltaSigma_1h + DeltaSigma_2h
        elif inclusion_type == "max":
            return np.maximum(DeltaSigma_1h, DeltaSigma_2h)
