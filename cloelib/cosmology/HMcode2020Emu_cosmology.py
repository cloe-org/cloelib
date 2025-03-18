# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate
# General imports
import numpy as np
from typing import Tuple, Optional
from copy import deepcopy

# Cosmology imports
try:
    import HMcode2020Emu as hmcodeemu
    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = \
        HM2020_emu.emulator['linear']['bounds']['z'][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


"""

## Notes:

- Adapted from ABC classes

"""

class HMemuLinearPerturbations:
    def __init__(self, background : Background, redshifts: np.ndarray):

        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background

        self.params_hm_emu = {
            'omega_cdm': self.background.Omega_cdm0,
            'omega_baryon': self.background.Omega_b0,
            'As': self.background.As,
            'ns': self.background.ns,
            'hubble': self.background.H0 / 100,
            'neutrino_mass': 0.06,
            'w0': self.background.w0,
            'wa': self.background.wa,
        }

        hm_bounds = HM2020_emu.emulator['linear']['bounds']

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 lin emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

        self.params_hm_emu['z'] = self.z

        _, Pk = HM2020_emu.get_linear_pk(**self.params_hm_emu)

        k_emu = HM2020_emu.emulator['linear']['k'] * self.background.h

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = \
            extend_spectra(k_emu, self.z , self.background.h ** -3 * Pk,
                           flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)

        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(
            self.z , self.k, Pk_out, kx=1, ky=1)

        self.Pk_interp = pk_interp

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Computes the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """

        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


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
        if hasattr(self, 'Pk_interp') and self.Pk_interp is not None:
            D_z_k = np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        return D_z_k

    def growth_rate(self, zs) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        params = deepcopy(self.params_hm_emu)
        params['z'] = zs

        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**params)

        return self.fsigma8/self.sigma8

class HMemuNonLinearPerturbations:
    def __init__(self, background : Background,
                 linearperturbations: Perturbations, redshifts: np.ndarray,
                 log10TAGN: Optional[float] = None):

        assert background.Omega_k0 == 0, 'Non flat geometries not supported'

        redshift_max = \
            HM2020_emu.emulator['nonlinear']['bounds']['z'][1]

        self.z = redshifts[redshifts <= redshift_max]
        self.background = background

        self.params_hm_emu = {
            'omega_cdm': self.background.Omega_cdm0,
            'omega_baryon': self.background.Omega_b0,
            'As': self.background.As,
            'ns': self.background.ns,
            'hubble': self.background.H0 / 100,
            'neutrino_mass': 0.06,
            'w0': self.background.w0,
            'wa': self.background.wa,
        }
        baryonic_boost = (log10TAGN is not None)

        if baryonic_boost:
            self.params_hm_emu['log10TAGN'] = log10TAGN

        hm_bounds = HM2020_emu.emulator['nonlinear']['bounds']

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 NL emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key],
                                                  len(self.z))

        self.params_hm_emu['z'] = self.z

        _, Pk = HM2020_emu.get_nonlinear_pk(nonu=False,
                                            **self.params_hm_emu,
                                            baryonic_boost=baryonic_boost)

        k_emu = HM2020_emu.emulator['nonlinear']['k'] * self.background.h

        # Low-k extrapolation.
        # Done this way to use Pk array instead of calling an interpolator
        # This only works if the redshift array is exactly the same within
        # range. This should be, but we should probably make sure in some way
        Pk_lin_mask_k = linearperturbations.k < k_emu[0]
        Pk_lin_mask_z = linearperturbations.z <= redshift_max
        Pk_lin = linearperturbations.Pk[Pk_lin_mask_z][:,Pk_lin_mask_k]
        k_all = np.concatenate((linearperturbations.k[Pk_lin_mask_k], k_emu))
        Pk_all = np.concatenate((Pk_lin, self.background.h ** -3 * Pk),axis=1)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = \
            extend_spectra(k_all, self.z, Pk_all,
                           flag_range=True,
                           option_wavenumber="power_law",
                           option_redshift="power_law", extrap_z = redshifts,
                           option_cosmo="const", ns=self.background.ns)

        # Aleternative method using interpolators
        # Pk_lin = linearperturbations.Pk_interp(self.z, k_emu)
        # k_out, z_out, boost_out = \
        #     extend_spectra(k_emu, self.z, self.background.h ** -3 * Pk / Pk_lin,
        #                    flag_range=True,
        #                    option_wavenumber="power_law",
        #                    option_redshift="power_law", extrap_z = redshifts,
        #                    option_cosmo="const", ns=self.background.ns)
        # self.Pk = linearperturbations.Pk_interp(self.z, k_out) * boost_out

        self.k = k_out
        self.z  = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(
            self.z , self.k, self.Pk, kx=1, ky=1)

        self.Pk_interp = pk_interp


    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Computes the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """

        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


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
        if hasattr(self, 'Pk_interp') and self.Pk_interp is not None:
            D_z_k = np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))

        return D_z_k

    def growth_rate(self, zs) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        params = deepcopy(self.params_hm_emu)
        params['z'] = zs

        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**params)

        return self.fsigma8/self.sigma8
