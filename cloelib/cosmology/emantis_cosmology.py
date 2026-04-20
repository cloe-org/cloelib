"""Implementation of nonlinear Perturbation cosmology for f(R) gravity using the e-MANTIS emulator."""

import numpy as np
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.cosmology.cosmology import Background, Perturbations
from emantis.matter_power_spectrum import NonLinearMGBoostEmulator
from scipy import interpolate


class EmantisFofrNonLinearPerturbations:
    """Nonlinear perturbations in f(R) gravity.

    The nonlinear matter power spectrum boost, i.e. P_f(R)(k) / P_LCDM(k), is computed with the e-MANTIS emulator.
    The boost is then combined with an external nonlinear LCDM matter power spectrum prediction
    in order to obtain the full nonlinear matter power spectrum in f(R) gravity.
    """

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        nonlinearperturbations_lcdm: Perturbations,
        fR0: float,
        redshifts: np.ndarray,
        verbose: bool = False,
        emu_version: int = 2,
        extrapolate_cosmo: bool = True,
    ) -> None:
        """Initialize the perturbation instance.

        Parameters
        ----------
        background : Background
            A background cosmology object, providing all the standard cosmological parameters.
        linearperturbations : Perturbations
            A linear perturbations object, providing linear perturbations in f(R) gravity.
        nonlinearperturbations_lcdm : Perturbations
            A nonlinear perturbations object, providing the nonlinear matter power spectrum in LCDM.
        fR0 : float
            The modified gravity parameter fR0.
        redshifts : np.ndarray
            An array of redshifts used to compute the matter power spectrum.
        verbose : float, optional (default=False)
            Activate or not verbose output from the emantis emulator.
        emu_version : int, optional (default=2)
            The version of the emantis emulator to use.
            Version 2 (the default) has an extended cosmological parameter and redshift range.
        extrapolate_cosmo : bool optional (default=True)
            Activate or not constant extrapolation of cosmological parameters.
            The extrapolation is done only for the LCDM cosmological parameters.
            There is no extrapolation for fR0.
        """
        self.background = background
        self.linearperturbations = linearperturbations
        self.nonlinearpertubations_lcdm = nonlinearperturbations_lcdm

        # Check and process emulator version.
        if emu_version == 2:
            model = "fR"
        elif emu_version == 1:
            model = "fR_v1"
        else:
            raise ValueError(
                "Unsupported value for `emu_version`. Allowed values are 1 or 2."
            )

        # Initialize the e-MANTIS emulator.
        self.emantis_emu = NonLinearMGBoostEmulator(model=model, verbose=verbose)

        # Build parameter dict. for the emulator.
        # These are the parameters common to the v1 and v2 of the emulator.
        self.params_emu = {
            "Omega_m": self.background.Omega_cdm0
            + self.background.Omega_b0
            + self.background.mnu / 93.14 / self.background.h**2,
            "sigma8_lcdm": self.nonlinearpertubations_lcdm.sigma8_0(),
            "logfR0": -np.log10(np.abs(fR0)),
        }

        # Add extra parameters supported only by the v2 of the emulator.
        if emu_version == 2:
            self.params_emu["n_s"] = self.background.ns
            self.params_emu["h"] = self.background.h
            self.params_emu["Omega_b"] = self.background.Omega_b0

        # Compute scale factor values.
        aexp_values = 1 / (1 + redshifts)

        # Minimum scale factor and maximum redshift of the emulator.
        aexp_min = np.min(self.emantis_emu.aexp_nodes)
        z_max = 1 / aexp_min - 1

        # Select scale factor and redshift values within emulator range.
        z_emu = redshifts[redshifts <= z_max]
        aexp_emu = aexp_values[aexp_values >= aexp_min]

        # Get default wavenumber bins from the emulator (h/Mpc -> 1/Mpc).
        k_emu = self.emantis_emu.kbins * self.params_emu["h"]

        # Let the emulator do it's own extrapolation to low-k values.
        k_min = 0.0001
        k_emu = np.concatenate(
            [np.linspace(k_min, k_emu[0], 100, endpoint=False), k_emu]
        )

        # Get boost from emantis (pass wavenumbers in h/Mpc).
        pk_boost_emu = self.emantis_emu.predict_boost(
            self.params_emu,
            aexp_emu,
            k=k_emu / self.params_emu["h"],
            extrapolate_cosmo=extrapolate_cosmo,
        )

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

        # Get LCDM matter power spectrum.
        pk_lcdm = self.nonlinearpertubations_lcdm.matter_power_spectrum(self.z, self.k)

        # Build interpolation for full power spectrum.
        self.pk_interp = interpolate.RectBivariateSpline(
            self.z, self.k, pk_lcdm * pk_boost_extended
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Compute the nonlinear matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            Redshift values at which to compute the matter power spectrum.

        ks: numpy.ndarray
            Wavenumber values at which to compute the matter power spectrum, in units of h/Mpc.

        Returns
        -------
        pk: numpy.ndarray
            Nonlinear matter power spectrum at the input redshift and wavenumber values.
        """
        return self.pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Compute the growth factor for some input redshift and wavenumber values.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters:
        -----------
        zs : array_like
            Redshift values at which to compute the growth factor.
        ks : array_like
            Wavenumber values at which to compute the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor for the input redshift and wavenumber values.
        """
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks) / self.matter_power_spectrum(0, ks)
        )

        return D_z_k

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """

        return self.linearperturbations.sigma8_0()
