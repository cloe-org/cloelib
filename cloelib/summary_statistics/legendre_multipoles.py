# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.observables.spectro import SpectroPower
from cloelib.auxiliary.math_utils import legendre

# General imports
from typing import Protocol, Union, TypeVar, Optional, Generic
import numpy as np
import jax.numpy as jnp
from scipy import integrate
from scipy.special import roots_legendre


class LegendreMultipoles:
    r"""Class to compute spectroscopic Legendre multipoles of the galaxy
    power spectrum using an external non-linear code

    Parameters
    ----------
    spectro_power: SpectroPower
        Class returning the anisotropic power spectrum (only density and
        velocity field couplings; noise and systematics are included directly
        here)
    background_fiducial: Background
        Background class for computing fiducial background distances
    parameters: dict
        Dictionary containing shot noise and parameters related to
        observational systematics
    nbar: float
        Mean number denisty of the sample
    """

    def __init__(self, spectro_power: SpectroPower,
                 background_fiducial: Background,
                 parameters: dict,
                 nbar: float):
        r"""Class constructor
        """

        self.spectro_power = spectro_power
        self.redshift = spectro_power.redshift

        self.background_fiducial = background_fiducial

        mu_min = 0.0
        mu_max = 1.0
        mu_samp = 101
        self.mu_grid = np.linspace(mu_min, mu_max, mu_samp)

        self.parameters = parameters
        self.nbar = nbar


    def _q_AP_tr(self, zs: np.ndarray) -> np.ndarray:
        r"""AP distortion parameter transversal to the line of sight
        .. math::
            q_{\perp}(z) &= \frac{D_{\rm M}(z)}{D_{\rm M,fid}(z)}\\
        Parameters
        ----------
        z: np.ndarray
           Redshift
        Returns
        -------
        q_tr: np.ndarray
           Transversal AP parameter
        """
        return (self.spectro_power.background.angular_diameter_distance(zs)
                / self.background_fiducial.angular_diameter_distance(zs))

    def _q_AP_lo(self, zs: np.ndarray) -> np.ndarray:
        r"""AP distortion parameter parallel to the line of sight
        .. math::
            q_{\parallel}(z) &= \frac{H_{\rm fid}(z)}{H(z)}\\
        Parameters
        ----------
        z: np.ndarray
           Redshift
        Returns
        -------
        q_tr: np.ndarray
           Parallel AP parameter
        """
        return (self.background_fiducial.hubble_parameter(zs)
                /self.spectro_power.background.hubble_parameter(zs))

    def _ensure_array(self, param):
        if np.isscalar(param):
            param = np.array([param])
        return np.asarray(param)

    def _k_AP(self, k: np.ndarray, mu: np.ndarray, zs: float,
              use_AP: Optional[bool] = True) -> np.ndarray:
        r"""AP-distorted wavenumber
        .. math::
            k(k_{\rm fid},\mu_{\rm fid}, z) &= k_{\rm fid} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{1/2}
        Parameters
        ----------
        k: np.ndarray
           Fiducial wavenumber
        mu: np.ndarray
           Fiducial angle (cosinus) to the line of sight
        z: float
           Redshift
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        kAP: np.ndarray
           AP-distorted wavenumber
        """
        q_tr = self._q_AP_tr(zs) if use_AP else 1.0
        q_lo = self._q_AP_lo(zs) if use_AP else 1.0
        return np.outer(k, np.sqrt(mu**2 / q_lo**2 + (1.0-mu**2) / q_tr**2))

    def _mu_AP(self, mu: np.ndarray, zs: float,
               use_AP: Optional[bool] = True) -> np.ndarray:
        r"""AP-distorted angle (cosinus) to the line of sight
        .. math::
            \mu(\mu_{\rm fid}, z) &= \frac{\mu_{\rm fid}}{q_\parallel(z)} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{-1/2}
        Parameters
        ----------
        mu: np.ndarray
           Fiducial angle (cosinus) to the line of sight
        z: float
           Redshift
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        muAP: np.ndarray
           AP-distorted angle (cosinus) to the line of sight
        """
        q_tr = self._q_AP_tr(zs) if use_AP else 1.0
        q_lo = self._q_AP_lo(zs) if use_AP else 1.0
        return mu / q_lo / np.sqrt(mu**2 / q_lo**2 + (1.0-mu**2) / q_tr**2)

    def _damping_function(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Damping function due to GCsp redshift uncertainty
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        damping_function: np.ndarray
            Damping function due to GCsp redshift uncertainty
        """
        sigma_z = self.parameters['sigmaz']
        sigma_r = 299792.458 * sigma_z / \
            self.background_fiducial.hubble_parameter(self.redshift)
        return np.exp(-k**2 * mu**2 * sigma_r**2)

    def _Pk2d_noise(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from expansion of stochastic field
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        Pk2d_noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = self.parameters['NP0'] + k**2 * (self.parameters['NP20'] +
                                                 self.parameters['NP22'] *
                                                 legendre(2, mu))
        return noise / self.nbar

    def _Pk2d_tot(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Total 2D power spectrum (including RSD, systematics, and noise)
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        Pk2d_tot: np.ndarray
            Total 2D power spectrum (including RSD, systematics, and noise)
        """
        return (self.spectro_power.Pk2d_rsd(k, mu) *
                self._damping_function(k, mu) *
                (1.0 - self.parameters['fout'])**2 +
                self._Pk2d_noise(k, mu))

    def power_multipoles(self, k: np.ndarray,
                         ells: Optional[np.ndarray] = None,
                         use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        ells = self._ensure_array(ells) if ells else np.array([0,2,4])
        AP_factor = (self._q_AP_tr(self.redshift)**2 *
                     self._q_AP_lo(self.redshift) if use_AP else 1.0)
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / \
            AP_factor
        multipoles = {}
        for i,ell in enumerate(ells):
            multipoles[f'ell{ell}'] = \
                integrate.simps(self._Pk2d_tot(self._k_AP(k, self.mu_grid,
                                                          self.redshift,
                                                          use_AP=use_AP),
                                               self._mu_AP(self.mu_grid,
                                                           self.redshift,
                                                           use_AP=use_AP)) *
                                legendre(ell, self.mu_grid),
                                self.mu_grid, axis=1)
            multipoles[f'ell{ell}'] *= (2.0 * prefactors[i])
        return multipoles

    def convolved_power_multipoles(self, parameters: dict, mixing_matrix=dict):
        r"""Power spectrum Legendre multipoles convolved with the mixing matrix
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        multipoles_out: dict
            Convolved power spectrum Legendre multipoles
        """

        self.mixing_matrix_dict = mixing_matrix

        for key in self.mixing_matrix_dict:
            self.mixing_matrix_dict[key] = \
                np.asarray(self.mixing_matrix_dict[key], dtype=np.float32)

        kin0 = self.mixing_matrix_dict['kin0']
        kin2 = self.mixing_matrix_dict['kin2']
        kin4 = self.mixing_matrix_dict['kin4']
        kout = self.mixing_matrix_dict['kout']

        multipoles_in = {}
        if np.all(kin0 == kin2) and np.all(kin2 == kin4):
            multipoles_in = self.power_multipoles(k=kin0,
                                                  parameters=parameters,
                                                  ells=[0,2,4])
        else:
            for ell in [0,2,4]:
                multipoles_in[f'ell{ell}'] = \
                    self.power_multipoles(
                        k=self.mixing_matrix_dict[f'kin{ell}'],
                        parameters=parameters, ells=[ell])

        for key in multipoles_in:
            multipoles_in[key] = \
                np.asarray(multipoles_in[key], dtype=np.float32)

        multipoles_out = {}
        multipoles_out['k'] = kout
        for ell in [0, 2, 4]:
            multipoles_out[f'ell{ell}'] = np.zeros(kout.shape)
            for ell_prime in [0, 2, 4]:
                multipoles_out[f'ell{ell}'] += \
                    np.dot(self.mixing_matrix_dict[f'W{ell}{ell_prime}'],
                           multipoles_in[f'ell{ell_prime}'])

        return multipoles_out
