"""Implementation of Background and Perturbation cosmology using EuclidEmulator2."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate

# General imports
import numpy as np
from typing import Sequence

# Cosmology imports
try:
    import euclidemu2

    ee2 = euclidemu2.PyEuclidEmulator()

except ImportError:
    raise ImportError("EuclidEmulator2 could not be imported or initialised.")


class EE2NonLinearPerturbations:
    """Class for nonlinear perturbations using EE2, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
    ):
        """Initialize the EE2NonLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = ee2.z_max
        self.z = redshifts[redshifts <= redshift_max]

        self.background = background
        self.linearperturbations = linearperturbations

        hubble = self.background.H0 / 100

        self.params_ee2 = {
            "Omega_b": self.background.Omega_b0,
            "Omega_m": self.background.Omega_m(0),
            "h": hubble,
            "A_s": self.background.As,
            "n_s": self.background.ns,
            "m_ncdm": _set_neutrino_masses(self.background),
            "w0_fld": self.background.w0,
            "wa_fld": self.background.wa,
        }

        ee2_bounds = ee2.bounds

        # At the moment we raise an error when out of range. May decide to extrapolate later
        for key in self.params_ee2.keys():
            if np.prod(self.params_ee2[key] - np.array(ee2_bounds[key])) > 0:
                raise ValueError("EE2 out of range.")
            else:
                continue

        k_emu, boost = ee2.get_boost(self.params_ee2, self.z)

        k_emu = k_emu * hubble

        boost_arr = np.array([boost[i] for i in range(len(self.z))])

        # Here only the method using interpolators will work in general
        k_out, z_out, boost_out = extend_spectra(
            k_emu,
            self.z,
            boost_arr,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.boost_interp = interpolate.RectBivariateSpline(
            z_out, np.log(k_out), boost_out, kx=1, ky=1
        )

        # outputed k is different from k_out above to improve k sampling when
        # later using the interpolator above for the C_ell calculation.
        # This may be changed if C_ell calculation is modified.
        self.k = linearperturbations.k
        self.z = z_out

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Compute the total matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            Total matter power spectrum at the specified scale
            and redshift

        """
        return self.boost_interp(
            zs, np.log(ks)
        ) * self.linearperturbations.matter_power_spectrum(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        r"""Compute the CDM+baryons power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            CDM+baryons power spectrum at the specified scale
            and redshift

        """

        # I use the approximation (used e.g. in Bacco) that
        # neutrinos are linear and P_{m\nu} is replaced by linear calculation.
        Pcb_L = self.linearperturbations.matter_power_spectrum_cb(zs, ks)
        Pmm_L = self.linearperturbations.matter_power_spectrum(zs, ks)
        boost = self.boost_interp(zs, np.log(ks))
        f_cb = (
            self.background.Omega_cdm0 + self.background.Omega_b0
        ) / self.background.Omega_m(0)

        return Pcb_L + (boost - 1) * Pmm_L / f_cb**2

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        We use here the growth from the fluctuations without baryons.

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

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        We use here the growth from the fluctuations without baryons.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        return self.linearperturbations.growth_rate()

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """

        return self.linearperturbations.sigma8_0()


def _set_neutrino_masses(background: Background) -> float:
    r"""Set neutrino masses in the parameters dictionary.

    This method adds neutrino masses to the provided dictionary.
    It also ensures consistency with the background cosmology.
    EE2 only supports a 3 degenerate massive neutrinos, but no error
    thrown, it will just assume take the total mass and use that,
    as this should be a small effect on the boost.

    Parameters
    ----------
    background: Background
        Background class containing cosmology and background quantities

    Returns
    -------
    float
        The total neutrino mass in eV.
    """

    if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
        raise ValueError(
            "EE2 only supports a fixed number of effective"
            f"relativistic species (N_eff=3.044). Found {background.N_eff} "
            "Ensure that N_eff=3.044 in the Background class."
        )
    if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
        mnu_arg = np.sum(np.asarray(background.mnu))
    else:
        mnu_arg = float(background.mnu)
    # returns the neutrino mass in eV
    return mnu_arg
