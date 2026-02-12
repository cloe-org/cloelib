import numpy as np
from scipy.integrate import simpson
from scipy.stats import skewnorm

from cloelib.auxiliary import units
from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.auxiliary import (
    convert_distance,
    convert_to_Delta_crit,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class HaloProfileCore:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        z=np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        zs_max: float = 4.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
    ):
        r"""Auxiliary class computing quantities used in mass profile models.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        matter_statistics : MatterStatistics
            An object from the `MatterStatistics` class.
        """
        self.matter_statistics = matter_statistics
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

        # ???
        self.z = z
        self.zs_max = zs_max
        self.mean_nz = mean_nz
        self.sigma_nz = sigma_nz
        self.alpha_nz = alpha_nz

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.z)
        self.nzs = self.n_zs(self.z)

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.matter_statistics.background

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
        d_a_sources = self.matter_statistics.angular_diameter_distance(z_sources)  # Mpc
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_a_lens = self.matter_statistics.angular_diameter_distance(z)[
            :, np.newaxis
        ]  # Mpc
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
            z_s = np.linspace(_z + 1.0e-10, self.zs_max, len(self.z))
            n_zs[z_ind] = skewnorm.pdf(
                z_s,
                self.alpha_nz,
                self.mean_nz,
                self.sigma_nz,
            )

        return n_zs

    def sigma_crit_inv_eff(self, z, zbin):
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
        # z_s is temporarily hard-coded
        z_s = np.linspace(z + 1.0e-10, self.zs_max, len(self.z), axis=1)
        sig_crit_m1 = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s)

        return self.nzsnorM[zbin] * simpson(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def surface_mass_density_args(self, R, z, M, radius_units="Mpc/h"):
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
            convert_to_Delta_crit(
                self.overdensity_type, self.overdensity, self.background, z
            )
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2.0
        )[:, np.newaxis, np.newaxis]

        RDelta = (
            3.0 * M[np.newaxis, :, np.newaxis] / 4.0 / np.pi / densityThreshold
        ) ** (1.0 / 3.0)

        if radius_units.lower() != "mpc/h":
            D_A = (
                self.matter_statistics.angular_diameter_distance(z) * self.background.h
            )  # Mpc / h
            R_outshape = convert_distance(R, radius_units, "Mpc/h", D_A[:, np.newaxis])[
                :, np.newaxis
            ]
        else:
            R_outshape = R[np.newaxis, np.newaxis, :]

        return R_outshape, RDelta, densityThreshold

    def check_profile_shape(self, R, z, M, profile):
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

    def _include_2h_term(
        self, inclusion_type, term_1h, func_2h, R, z, halo_bias, radius_units
    ):
        """
        Computes and adds the 2-halo term.

        Parameters
        ----------
        term_1h : np.ndarray
            1 halo term, with shape (z.size, M.size, R.size).
        func_2h : function
            Function that computes the 2h term. It must take (R, z, radius_units)
            inputs, and output shape (z.size, R.size)
        inclusion_type : str
            If "sum", the 1-halo and 2-halo profile are summed.
            If "max", the maximum between them is considered at each point.
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        halo_bias: np.ndarray
            Halo bias, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
        """
        if inclusion_type not in ("sum", "max"):
            raise ValueError(
                "Invalid 'inclusion_type' definition, %s." % inclusion_type
            )

        # Check 2h inputs
        if halo_bias is None:
            raise ValueError(
                "halo_bias must be provided explicitly when computing the 2-halo term."
            )

        if halo_bias.shape != term_1h.shape[:2]:
            raise ValueError(
                f"halo_bias shape {halo_bias.shape} must "
                f"be the same as first two of term_1h {term_1h.shape}"
            )

        # compute and add 2h term
        term_2h = (
            func_2h(R, z, radius_units)[:, np.newaxis, :] * halo_bias[:, :, np.newaxis]
        )

        if inclusion_type == "sum":
            return term_1h + term_2h
        elif inclusion_type == "max":
            return np.maximum(term_1h, term_2h)

    def include_surface_mass_density_2h(
        self, Sigma_1h, inclusion_type, R, z, halo_bias, radius_units="Mpc/h"
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
        return self._include_2h_term(
            inclusion_type,
            Sigma_1h,
            func_2h=self.matter_statistics.surface_mass_density_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )

    def include_excess_surface_mass_density_2h(
        self, DeltaSigma_1h, inclusion_type, R, z, halo_bias, radius_units="Mpc/h"
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
        return self._include_2h_term(
            inclusion_type,
            DeltaSigma_1h,
            func_2h=self.matter_statistics.excess_surface_mass_density_2h,
            R=R,
            z=z,
            halo_bias=halo_bias,
            radius_units=radius_units,
        )
