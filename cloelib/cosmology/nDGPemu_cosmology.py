"""Implementation of nonlinear Perturbation cosmology for nDGP gravity using the nDGPemu emulator."""

import numpy as np
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.cosmology.cosmology import Background, Perturbations
from nDGPemu import BoostPredictor
from scipy import interpolate


class NDGPemuNonLinearPerturbations:
    """Nonlinear perturbations in nDGP.

    The nonlinear matter power spectrum boost, i.e. P_nDGP(k) / P_LCDM(k), is computed with the nDGPemu emulator.
    The boost is then combined with an external nonlinear LCDM matter power spectrum prediction
    in order to obtain the full nonlinear matter power spectrum.
    """

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        nonlinearperturbations_lcdm: Perturbations,
        omega_rc: float,
        redshifts: np.ndarray,
        extrapolate_cosmo: bool = True,
    ) -> None:
        """Initialize the perturbation instance.

        Parameters
        ----------
        background : Background
            A background cosmology object, providing all the standard cosmological parameters.
        linearperturbations : Perturbations
            A linear perturbations object, providing linear perturbations in nDPG.
        nonlinearperturbations_lcdm : Perturbations
            A nonlinear perturbations object, providing the nonlinear matter power spectrum in LCDM.
        omega_rc : float
            The modified gravity parameter omega_rc.
        redshifts : np.ndarray
            An array of redshifts used to compute the matter power spectrum.
        extrapolate_cosmo : bool optional (default=True)
            Activate or not constant extrapolation of cosmological parameters.
            The extrapolation is done only for the LCDM cosmological parameters.
            There is no extrapolation for omega_rc.
        """
        self._background = background
        self.linearperturbations = linearperturbations
        self.nonlinearpertubations_lcdm = nonlinearperturbations_lcdm

        # Initialize nDGPemu.
        self.ndgpemu = BoostPredictor()

        # Cosmological parameter ranges of the emulator.
        param_ranges = {
            "Om": [0.28, 0.36],
            "Ob": [0.04, 0.06],
            "ns": [0.92, 1],
            "As": [1.7e-9, 2.5e-9],
            "h": [0.61, 0.73]
        }

        # Build parameter dict. of standard parameters for the emulator.
        self.params_emu = {
            "Om": self.background.Omega_m(0),
            "Ob": self.background.Omega_b(0),
            "ns": self.background.ns,
            "As": self.background.As,
            "h": self.background.h,
        }

        # If activated, constant extrapolation in cosmological parameters (not omega_rc).
        if extrapolate_cosmo:
            for param in self.params_emu:
                self.params_emu[param] = np.clip(self.params_emu[param], a_min=param_ranges[param][0], a_max=param_ranges[param][1])

        # nDGP parameter for the emulator.
        H0rc = (1 / 4 / omega_rc)**(1/2)

        # Maximum redshift of the emulator.
        z_max = 2

        # Select scale factor and redshift values within emulator range.
        z_emu = redshifts[redshifts <= z_max]

        # Get default wavenumber bins from the emulator (h/Mpc -> 1/Mpc).
        k_emu = self.ndgpemu.k_vals * self.params_emu["h"]

        # Let the emulator do it's own extrapolation to low-k values.
        k_min = 0.0001
        k_emu = np.concatenate(
            [np.linspace(k_min, k_emu[0], 100, endpoint=False), k_emu]
        )

        # Get boost from nDGPemu.
        # Loop over redshifts, with a call to the emulator each time.
        pk_boost_emu = np.zeros((len(z_emu), len(k_emu)))
        for i,z_val in enumerate(z_emu):
            pk_boost_emu[i,:] = self.ndgpemu.predict(H0rc, z_val, self.params_emu, k_out=k_emu / self.params_emu["h"], ext=0)

        # Extrapolate emulator prediction in wavenumber and redshift.
        k_extended, z_extended, pk_boost_extended = extend_spectra(
            k_emu,
            z_emu,
            pk_boost_emu,
            flag_range=True,
            option_wavenumber="const",
            option_redshift="power_law",
            option_cosmo="const",
            extrap_z=redshifts,
            ns=self.background.ns,
        )

        self.k = k_extended
        self.z = z_extended

        # Build interpolation for power spectrum boost.
        self.boost_interp = interpolate.RectBivariateSpline(
            self.z, np.log(self.k), pk_boost_extended
        )

    @property
    def background(self) -> Background:
        """Return the background object."""
        return self._background

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Compute the nonlinear total matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            Redshift values.

        ks: numpy.ndarray
            Wavenumber values in units of h/Mpc.

        Returns
        -------
        pk: numpy.ndarray
            Nonlinear total matter power spectrum at the input redshift and wavenumber values.
        """
        return self.boost_interp(
            zs, np.log(ks)
        ) * self.nonlinearpertubations_lcdm.matter_power_spectrum(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        """Compute the nonlinear CDM+baryons power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            Redshift values.

        ks: numpy.ndarray
            Wavenumber values in units of h/Mpc.

        Returns
        -------
        pk: numpy.ndarray
            Nonlinear CDM+baryons power spectrum at the input redshift and wavenumber values.
        """

        # We are assuming that we can apply the same MG boost to the total matter
        # and to the CDM+baryons matter power spectra.
        # In any case, nDGPemu has been calibrated without neutrinos, so any use of neutrinos
        # (or other non-cold matter) with this perturbation class will require some testing.
        return self.boost_interp(
            zs, np.log(ks)
        ) * self.nonlinearpertubations_lcdm.matter_power_spectrum_cb(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Compute the growth factor D(z, k) normalized to D(0).

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """

        return self.linearperturbations.growth_factor(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """Compute the growth rate f(z, k) for given redshifts and wavenumbers.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        return self.linearperturbations.growth_rate(zs, ks)

    def sigma8_0(self) -> float:
        """
        Compute the sigma8 value for the current cosmology at z=0.

        Returns:
        --------
        float
            The sigma8 value.
        """

        return self.linearperturbations.sigma8_0()
