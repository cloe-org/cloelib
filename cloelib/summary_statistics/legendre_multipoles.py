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
        noise = (self.parameters['NP0'] * self._Pk2d_noise_k0(k)
                 + self.parameters['NP20'] * self._Pk2d_noise_k0(k)
                 + self.parameters['NP22'] * self._Pk2d_noise_k2mu2(k, mu))
        return noise

    def _Pk2d_noise_k0(self, k: np.ndarray) -> np.ndarray:
        r"""Leading-order term from 2d power spectrum of stochastic field
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = np.full_like(k, 1.0)
        return noise / self.nbar

    def _Pk2d_noise_k2(self, k: np.ndarray) -> np.ndarray:
        r"""Isotropic next-to-leading-order term from 2d power spectrum of
        stochastic field
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2
        return noise / self.nbar

    def _Pk2d_noise_k2mu2(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Anisotropic next-to-leading-order term from 2d power spectrum of
        stochastic field
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2 * legendre(2, mu)
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
        ells = self._ensure_array(ells) if ells is not None else np.array([0,2,4])
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

    def power_term_multipoles(self, k: np.ndarray, term_list: list,
                              ells: Optional[np.ndarray] = None,
                              use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        term_list: list
            List of terms to compute
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        ells = self._ensure_array(ells) \
            if ells is not None else np.array([0,2,4])
        AP_factor = (self._q_AP_tr(self.redshift)**2 *
                     self._q_AP_lo(self.redshift) if use_AP else 1.0)
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / \
            AP_factor
        kAP = self._k_AP(k, self.mu_grid, self.redshift, use_AP=use_AP)
        muAP = self._mu_AP(self.mu_grid, self.redshift, use_AP=use_AP)
        Pk2d = np.empty((len(term_list), len(k), len(self.mu_grid)))
        rsd_ids = [index for index, term in enumerate(term_list)
                   if 'noise' not in term]
        if rsd_ids:
            Pk2d[rsd_ids] = self.spectro_power.Pk2d_term_rsd(
                kAP, muAP, term_list=[term_list[index] for index in rsd_ids])
        noise_ids = [index for index in range(len(term_list))
                     if index not in rsd_ids]
        if noise_ids:
            noise_func = {'noise_k0': self._Pk2d_noise_k0,
                          'noise_k2': self._Pk2d_noise_k2,
                          'noise_k2mu2': self._Pk2d_noise_k2mu2}
            Pk2d[noise_ids] = np.array([
                noise_func[term_list[index]](kAP, muAP)
                if term_list[index]=='noise_k2mu2'
                else noise_func[term_list[index]](kAP) for index in noise_ids])
        multipoles = {}
        for i,ell in enumerate(ells):
            multipoles[f'ell{ell}'] = \
                integrate.simps(Pk2d * legendre(ell, self.mu_grid),
                                self.mu_grid, axis=-1)
            multipoles[f'ell{ell}'] *= (2.0 * prefactors[i])
        return multipoles

    def convolved_power_multipoles(self, mixing_matrix: dict,
                                   ells: Optional[np.ndarray] = None,
                                   use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles convolved with the mixing matrix
        Parameters
        ----------
        mixing_matrix: dict
            Dicitonary containing the mixing matrix
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles_out: dict
            Convolved power spectrum Legendre multipoles
        """
        ells_tot = [0, 2, 4]
        ells = self._ensure_array(ells) if ells is not None else ells_tot

        kin_arrays = [mixing_matrix[f'kin{ell}'] for ell in ells_tot]

        if all(np.array_equal(kin_arrays[0], kin) for kin in kin_arrays):
            multipoles_in = self.power_multipoles(
                k=kin_arrays[0], ells=ells_tot, use_AP=use_AP)
        else:
            multipoles_in = {
                f'ell{ell}': self.power_multipoles(k=kin_arrays[i], ells=[ell],
                                                   use_AP=use_AP)
                for i, ell in enumerate(ells_tot)}

        multipoles_out = {}
        multipoles_out['k'] = mixing_matrix['kout']
        for ell in ells:
            multipoles_out[f'ell{ell}'] = (
                sum(np.dot(mixing_matrix[f'W{ell}{ell_prime}'],
                           multipoles_in[f'ell{ell_prime}'])
                    for ell_prime in ells_tot))

        return multipoles_out
