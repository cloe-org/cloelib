# cloelite imports
from cloelite.cosmology.cosmology import Background
from cloelite.auxiliary.extrapolator import extend_spectra

from scipy import interpolate
# General imports
import numpy as np

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
            'neutrino_mass': 0.0,
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
        self.Pk = Pk

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
        # Understand this below
        k_emu = HM2020_emu.emulator['linear']['k'] * self.background.h

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = \
            extend_spectra(k_emu, self.z , self.background.h ** -3 * self.Pk,
                           flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law",
                           option_cosmo="const", ns=self.background.ns)

        self.k = k_out
        self.z  = z_out

        pk_linear = interpolate.RectBivariateSpline(
            self.z , self.k, Pk_out, kx=1, ky=1)
        
        
        self.Pk_linear = pk_linear

        return pk_linear(zs, ks)

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
        if hasattr(self, 'Pk_linear') and self.Pk_linear is not None:
            D_z_k = np.sqrt(self.Pk_linear(zs, ks) / self.Pk_linear(0.0, ks))
        else:
            self.matter_power_spectrum(zs, ks)
            D_z_k = np.sqrt(self.Pk_linear(zs, ks) / self.Pk_linear(0.0, ks))

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**self.params_hm_emu)

        return self.fsigma8/self.sigma8

class HMemuNonLinearPerturbations:
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
            'neutrino_mass': 0.0,
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

        _, Pk = HM2020_emu.get_nonlinear_pk(nonu=False, 
                                            **self.params_hm_emu,
                                            baryonic_boost=False)
        self.Pk = Pk


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
        # Understand this below


        k_emu = HM2020_emu.emulator['nonlinear']['k'] * self.background.h

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = \
            extend_spectra(k_emu, self.z , self.background.h ** -3 * self.Pk,
                           flag_range=True,
                           option_wavenumber="logk2",
                           option_redshift="power_law",
                           option_cosmo="const", ns=self.background.ns)

        self.k = k_out
        self.z  = z_out

        pk_nonlinear = interpolate.RectBivariateSpline(
            self.z , self.k, Pk_out, kx=1, ky=1)
        
        
        self.Pk_nonlinear = pk_nonlinear

        return pk_nonlinear(zs, ks)

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
        if hasattr(self, 'Pk_nonlinear') and self.Pk_nonlinear is not None:
            D_z_k = np.sqrt(self.Pk_nonlinear(zs, ks) / self.Pk_nonlinear(0.0, ks))
        else:
            self.matter_power_spectrum(zs, ks)
            D_z_k = np.sqrt(self.Pk_nonlinear(zs, ks) / self.Pk_nonlinear(0.0, ks))

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        self.sigma8, self.fsigma8 = HM2020_emu.get_sigma8(**self.params_hm_emu)

        return self.fsigma8/self.sigma8
