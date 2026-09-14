# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.derived_cosmology import growth_function_ODE

import numpy as np


class SplitLinearPerturbations:
    """Class to output the rescaled linear matter power spectrum for the
    growth-geometry split"""

    def __init__(
        self,
        background: Background,
        Omega_m_growth: float,
        redshifts: np.ndarray,
        lin_perturbations: Perturbations,
    ):
        """Initialise SplitLinearPerturbations."""
        self.background = background
        self.Omega_m_growth = Omega_m_growth
        self.z = redshifts
        self.kmax = 100
        self.lin_perturbations = lin_perturbations

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the growth-geometry split linear matter power
        spectrum. This implies a rescaling with the growth function of the
        matter power spectrum and also sigma_8, as in 2301.03694

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_linear: numpy.ndarray
            Rescaled linear matter power spectrum at the specified scale
            and redshift.
        """
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        pk_linear_EBS = self.lin_perturbations.matter_power_spectrum(zs, ks)  # type:ignore[union-attr]

        Omega_m_geo = self.background.Omega_cdm0 + self.background.Omega_b0

        # Compute the growth factor
        g_z_geo = growth_function_ODE(self.background, zs, Omega_m_geo)
        g_z_growth = growth_function_ODE(self.background, zs, self.Omega_m_growth)

        self.pk_linear = np.zeros_like(pk_linear_EBS)

        # Rescale the matter power spectrum with G(z)
        for i in range(len(zs)):
            rescale_fac = g_z_growth[i] ** 2 / g_z_geo[i] ** 2
            self.pk_linear[i, :] = rescale_fac * pk_linear_EBS[i, :]

        # Rescale sigma_8 here to avoid repeatedly calling the ODE
        self.sigma8 = g_z_growth[0] / g_z_geo[0] * self.lin_perturbations.sigma8_0()
        return self.pk_linear

    def matter_power_spectrum_cb(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the growth-geometry split linear matter power
        spectrum of cold dark matter + baryons (no neutrinos). This implies a
        rescaling with the growth function of the matter power spectrum and also
        sigma_8, as in 2301.03694

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_linear_cb: numpy.ndarray
            Rescaled linear matter power spectrum of cold dark matter + baryons
            at the specified scale and redshift.
        """
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")

        pk_linear_EBS_cb = self.lin_perturbations.matter_power_spectrum_cb(zs, ks)  # type:ignore[union-attr]

        Omega_m_geo = self.background.Omega_cdm0 + self.background.Omega_b0

        # Compute the growth factor
        g_z_geo = growth_function_ODE(self.background, zs, Omega_m_geo)
        g_z_growth = growth_function_ODE(self.background, zs, self.Omega_m_growth)

        self.pk_linear_cb = np.zeros_like(pk_linear_EBS_cb)

        # Rescale the matter power spectrum with G(z)
        for i in range(len(zs)):
            rescale_fac = g_z_growth[i] ** 2 / g_z_geo[i] ** 2
            self.pk_linear_cb[i, :] = rescale_fac * pk_linear_EBS_cb[i, :]

        # Rescale sigma_8 here to avoid repeatedly calling the ODE
        self.sigma8 = g_z_growth[0] / g_z_geo[0] * self.lin_perturbations.sigma8_0()
        return self.pk_linear_cb

    def growth_factor(
        self, zs, ks=np.logspace(np.log10(1e-5), np.log10(1e0), 200)
    ) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        d_z_k: numpy.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        d_z_k = np.zeros([len(zs), len(ks)])
        g_ode = growth_function_ODE(self.background, zs, self.Omega_m_growth)
        for i in range(len(ks)):
            d_z_k[:, i] = g_ode / (g_ode[0] * (1 + zs))

        return d_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        growth_rate_f: numpy.ndarray
            Scale-independent growth rate f(z)
        """
        growth_factor_D = self.growth_factor(self.z)[:, 0]

        derivative_growth_factor_D = np.gradient(growth_factor_D, self.z)

        growth_rate_f = -(1 + self.z) / growth_factor_D * derivative_growth_factor_D

        return growth_rate_f

    def sigma8_0(self) -> float:
        """
        Calculate the split sigma8 value. This is taken from the EBS and then
        rescaled with the growth as in 2301.03694, Equation (8).
        Only works if the matter_power_spectrum function has been called.
        This is to avoid repeatedly calling the ODE which takes time.

        Returns:
        --------
        sigma8: float
            The sigma8 value.
        """
        return self.sigma8


class SplitNonLinearPerturbations:
    """Class to output the rescaled non-linear matter power spectrum for the
    growth-geometry split"""

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        lin_perturbations_split: np.ndarray,
        lin_perturbations_growth: Perturbations,
        nl_perturbations_growth: Perturbations,
    ):
        """Initialize the OmgrowthLinearPerturbation and OmgrowthNonLinearPerturbation instance."""
        self.z = redshifts
        self.kmax = 100
        self.background = background
        self.lin_perturbations_split = lin_perturbations_split
        self.lin_perturbations_growth = lin_perturbations_growth
        self.nl_perturbations_growth = nl_perturbations_growth

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the split non-linear matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_nonlinear: numpy.ndarray
            Non-linear matter power spectrum at the specified scale
            and redshift
        """

        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        pk_linear_growth = self.lin_perturbations_growth.matter_power_spectrum(zs, ks)  # type:ignore[union-attr]
        pk_nonlinear_growth = self.nl_perturbations_growth.matter_power_spectrum(zs, ks)  # type:ignore[union-attr]

        # Compute the boost factor
        boost = pk_nonlinear_growth / pk_linear_growth

        # Multiply the boost to the rescaled power spectrum
        pk_nonlinear = boost * self.lin_perturbations_split.matter_power_spectrum(
            zs, ks
        )

        return pk_nonlinear.squeeze()

    def matter_power_spectrum_cb(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the split non-linear matter power spectrum of cold dark
        matter + baryons (no neutrinos).

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_nonlinear_cb: numpy.ndarray
            Non-linear matter power spectrum of cold dark matter + baryons at
            the specified scale and redshift
        """

        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        pk_linear_growth_cb = self.lin_perturbations_growth.matter_power_spectrum_cb(
            zs, ks
        )  # type:ignore[union-attr]
        pk_nonlinear_growth_cb = self.nl_perturbations_growth.matter_power_spectrum_cb(
            zs, ks
        )  # type:ignore[union-attr]

        # Compute the boost factor
        boost = pk_nonlinear_growth_cb / pk_linear_growth_cb

        # Multiply the boost to the rescaled power spectrum
        pk_nonlinear_cb = boost * self.lin_perturbations_split.matter_power_spectrum_cb(
            zs, ks
        )

        return pk_nonlinear_cb.squeeze()

    def growth_factor(
        self, zs, ks=np.logspace(np.log10(1e-5), np.log10(1e0), 200)
    ) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        d_z_k: numpy.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        d_z_k = np.zeros([len(zs), len(ks)])

        # Use the background of lin. pert. class, including Omega_m^growth

        bg = self.lin_perturbations_growth.background

        g_ode = growth_function_ODE(bg, zs, (bg.Omega_cdm0 + bg.Omega_b0))

        for i in range(len(ks)):
            d_z_k[:, i] = g_ode / (g_ode[0] * (1 + zs))

        return d_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        growth_rate_f: numpy.ndarray
            Scale-independent growth rate f(z)
        """
        growth_factor_D = self.growth_factor(self.z)[:, 0]

        derivative_growth_factor_D = np.gradient(growth_factor_D, self.z)

        growth_rate_f = -(1 + self.z) / growth_factor_D * derivative_growth_factor_D

        return growth_rate_f

    def sigma8_0(self) -> float:
        """
        Calculate the split sigma8 value. This is taken from the EBS and then
        rescaled with the growth as in 2301.03694, Equation (8).
        Only works if the matter_power_spectrum function has been called.
        This is to avoid repeatedly calling the ODE which takes time.

        Returns:
        --------
        sigma8_0: float
            The sigma8 value taken from the SplitLinearPerturbations class to
            account for the rescaling that is implemented in the split as in
            2301.03694, Equation (8).
        """
        return self.lin_perturbations_split.sigma8_0()
