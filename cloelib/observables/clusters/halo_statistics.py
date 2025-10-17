import numpy as np
from scipy import interpolate
from scipy.integrate import simpson as simps

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations


class HaloStatistics:
    def __init__(
        self,
        perturbations: Perturbations,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        nonu: bool = False,
        use_interpolation: bool = True,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""A class computing halo mass function and halo bias.

        Initialize the class with given perturbations and overdensity definition.

        Parameters
        ----------
        perturbations : Perturbations
            An object from the `LinearPerturbations` class containing cosmological
            perturbation data (e.g., power spectrum, growth function).
        overdensity_type : str
            Overdensity definition for halo mass calculation. Must be one of:
            - "crit": Relative to critical density of the universe.
            - "mean": Relative to mean matter density.
            - "vir": Virial overdensity from spherical collapse.
        overdensity : int, optional
            Value of the overdensity. Effective for non-virial overdensities.
            Example: If it equals 200, halos are defined as regions with density
            200 times the chosen reference (`crit` or `mean`).
        nonu : bool, optional
            If `True`, massive neutrinos are excluded from the density parameter
            summation.
        use_interpolation : bool, optional
            If true, class uses interpolation for matter power spectrum computation.
            A default interpolation is set when class is instanciated with
            use_interpolation=True. For a more customized interpolation, check
            the interpolate_matter_power_spectrum function.
        """
        self.perturbations = perturbations

        if overdensity_type not in ["crit", "mean", "vir"]:
            raise ValueError("Invalid overdensity definition, %s." % overdensity_type)
        self.overdensity_type = overdensity_type
        self.overdensity = overdensity

        self.nonu = nonu

        # Power spectrum interpolation
        self.Pk_interp = None

        # internal value of sigma8
        self.__sigma8 = None

        # set interpolation usage
        self.z = z
        self.k = k
        if use_interpolation:
            self.interpolate_matter_power_spectrum(self.z, self.k)
        self.use_interpolation = use_interpolation

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.perturbations.background

    @property
    def sigma8(self):
        r"""Returns the `sigma_8` value at redshift `z=0`. If `sigma_8`
        is not set as a base parameter, it is computed from the power spectrum.
        """
        if self.__sigma8 is None:
            self.__sigma8 = self.sigma_z_R([0.0], np.array([8.0]))
        return self.__sigma8

    @property
    def nonu(self):
        r"""Includes or not neutrinos on matter density and matter power spectrum."""
        return self.__nonu

    @nonu.setter
    def nonu(self, value):
        """Set nonu"""
        if not isinstance(value, bool):
            raise ValueError(f"value for nonu must be boolean, used {value}")
        self.__nonu = value
        if self.nonu:
            self._Omega_m = self.background.Omega_m_cb
            self._matter_power_spectrum = self.perturbations.matter_power_spectrum_cb
        else:
            self._Omega_m = self.background.Omega_m
            self._matter_power_spectrum = self.perturbations.matter_power_spectrum

    @property
    def use_interpolation(self):
        r"""If true, class uses interpolation for matter power spectrum computation."""
        return self.__use_interpolation

    @use_interpolation.setter
    def use_interpolation(self, use_interpolation):
        """If true, makes class uses interpolation for matter power spectrum computation."""
        if use_interpolation:
            self.matter_power_spectrum = self.Pk_interp
        else:
            self.matter_power_spectrum = _matter_power_spectrum_not_interpolated
        self.__use_interpolation = use_interpolation

    def _matter_power_spectrum_not_interpolated(self, z, k):
        r"""Computes the non interpolated matter power spectrum.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}

        Returns
        -------
        float or np.ndarray
            Matter power spectrum.
        """
        return self._matter_power_spectrum(
            z,
            k,
            hubble_units=True,
            k_hunit=True,
        )

    def interpolate_matter_power_spectrum(
        self,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""Create internal interpolation of matter power spectrum.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}
        """
        # Power spectrum interpolation
        self.Pk_interp = interpolate.RectBivariateSpline(
            z,
            k,
            self._matter_power_spectrum_not_interpolated(z, k),
        )

    def window(self, k, R):
        r"""Top-hat window and its derivative.

        Computes the top-hat window function and its derivative.

        Parameters
        ----------
        k: numpy.ndarray
               Wavenumber where W(kR) is evaluated.
               Units: h Mpc^{-1}
        R: numpy.ndarray
               Radius where W(kR) is evaluated.
               Units: h^{-1} Mpc

        Returns
        -------
        W: numpy.ndarray
            W[i,j], where i is the wavenumber axis and j the radius axis
        dWdx: numpy.ndarray
            dWdx[i,j], where i is the wavenumber axis and j the radius axis
        """
        x = R[:, np.newaxis] * k
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3.0
        dWdx = 3.0 * (np.sin(x) * (x**2.0 - 3.0) + 3.0 * x * np.cos(x)) / x**4.0

        return W, dWdx

    def radius_M(self, M):
        r"""Radius from a mass.

        Converts a mass into a radius.

        Parameters
        ----------
        M: numpy.ndarray
            Mass in h^{-1} Msun

        Returns
        -------
        radius_M: array
            Radius in h^{-1} Mpc
        """
        rho_m_0 = (
            derived_cosmology.rho_crit(self.background, 0.0)
            * self._Omega_m(0.0)
            / self.background.h**2.0
        )
        return (M / rho_m_0 * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

    def delta_c(self, z):
        r"""Critical overdensity.

        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999).

        Parameters
        ----------
        z: float or np.ndarray
            Redshift at which to evaluate the delta_c

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical overdensity a given redshift.
        """

        return (
            3.0
            / 20.0
            * (12.0 * np.pi) ** (2.0 / 3.0)
            * (1.0 + 0.012299 * np.log10(self.background.Omega_m(z)))
        )

    def get_Delta_crit(self, z):
        r"""Critical overdensity factor.

        Converts the input overdensity factor into a critical one.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.

        Returns
        -------
        overdensity: float or np.ndarray
            The overdensity factor which needs
            to be multiplied to the critical
            density in order to define an overdensity.

        Notes
        -----
        The function is returned for :math:`\rm \rho_c` in a density definition
        at a given redshift. The function returns :math:`\rm \Delta` for the
        critical density of the universe, :math:`\rm \Delta \Omega_{m}` for
        the mean matter density of the universe, :math:`\rm \Delta` determined
        by `Bryan & Norman 1998
        <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`_ Equation 6 for
        the virial density.
        """
        if self.overdensity_type == "crit":
            Delta = self.overdensity

        elif self.overdensity_type == "mean":
            Delta = self.overdensity * self._Omega_m(z)

        elif self.overdensity_type == "vir":
            x = self._Omega_m(z) - 1.0
            Delta = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

        return Delta

    def sigma_z_R(self, z, R):
        r"""Standard deviation of perturbations given a redshift and radius.

        Computes the rms at the radii requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        R: numpy.ndarray
            Radius points in h^{-1} Mpc.

        Returns
        -------
        sigma_z_R: numpy.ndarray
            sigma_z_R[i,j], where i is the redshift axis and j the radius axis.
        """
        k = self.k  # h/Mpc
        W, _ = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.matter_power_spectrum(z, k).reshape(len(z), 1, len(k))
                    * (W**2.0).reshape(1, len(R), len(k)),
                    x=k,
                    axis=-1,
                )
            )
        )

    def sigma_z_M(self, z, M):
        r"""Standard deviation of perturbations given a redshift and mass.

        Computes the rms at the masses requested from
        the table given by the Boltzman code.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        sigma_z_M: numpy.ndarray
            sigma_z_M[i,j], where i is the redshift axis and j the mass axis.
        """
        R = self.radius_M(M)  # Mpc/h
        return self.sigma_z_R(z, R)

    def nu_z_M(self, z, M):
        r"""Peak height.

        Computes the critical overdensity over the rms,
        delta_c/sigma, at a given redshift and mass.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points.

        Returns
        -------
        nu_z_M: numpy.ndarray
            nu_z_M[i,j], where i is the redshift axis and j the mass axis.
        """
        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnR(self, z, M):
        r"""Derivative of the logarithmic rms.

        Computes the derivative of the log rms
        with respect to the radius
        at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        dlns_dlnR: numpy.ndarray
            dlns_dlnR[i,j], where i is the redshift axis and j the mass axis.
        """
        k = self.k  # h/Mpc
        R = self.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        dsigma2_dR = np.pi**-2 * simps(
            k.reshape(1, 1, len(k)) ** 3
            * self.matter_power_spectrum(z, k).reshape(len(z), 1, len(k))
            * W.reshape(1, len(R), len(k))
            * dWdx.reshape(1, len(R), len(k)),
            x=k,
            axis=-1,
        )

        return R / (2 * self.sigma_z_M(z, M) ** 2) * dsigma2_dR
