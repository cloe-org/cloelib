import jax.numpy as np

from scipy import integrate
from scipy.special import legendre

from abc import ABC, abstractmethod

from typing import Optional

from cloelite.cosmology.cosmology import Background
from cloelite.cosmology.cosmology import LinearPerturbations

class LegendreMultipoles(ABC):

    def __init__(self, NLmodel: Optional[str] = 'EFT',
                 linear_perturbations: Optional[LinearPerturbations] = None,
                 background_fiducial: Optional[Background] = None):
        r"""Class constructor
        Parameters
        ----------
        NLcode: str
            Non-linear code used to compute the power spectrum
        NLmodel: str
            Non-linear model (only EFT supported for now)
        linear_perturbations: LinearPerturbations
            Linear perturbations
        background_fiducial: Background
            Fiducial background (for AP corrections)
        """
        mu_min = -1.0
        mu_max = 1.0
        mu_samp = 101
        self.mu_grid = np.linspace(mu_min, mu_max, mu_samp)

        if linear_perturbations:
            self.linear_perturbations = linear_perturbations
            self.background = linear_perturbations.background

        if background_fiducial:
            self.background_fiducial = background_fiducial

    def update(self, **kwargs):
        r"""Update method
        """
        self.linear_perturbations.update(**kwargs)
        self.background = self.linear_perturbations.background

    def set_number_density(self, nbar: float):
        r"""Setter of the number density
        Parameters
        ----------
        nbar: float
            Number density of the considered GCsp sample
        """
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
        return (self.background.angular_diameter_distance(zs)
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
        return (self.background_fiducial.hubble_parameter(zs, units='km/s/Mpc')
                / self.background.hubble_parameter(zs, units='km/s/Mpc'))

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

    @abstractmethod
    def _Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray,
                  parameters: dict) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_rsd: np.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        pass

    def _Pk2d_noise(self, k: np.ndarray, mu: np.ndarray,
                    parameters: dict) -> np.ndarray:
        r"""2D power spectrum from expansion of stochastic field
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = parameters['NP0'] + k**2 * (parameters['NP20'] +
                                            parameters['NP22'] *
                                            legendre(2)(mu))
        return noise / self.nbar

    def _damping_function(self, k: np.ndarray, mu: np.ndarray,
                          parameters: dict) -> np.ndarray:
        r"""Damping function due to GCsp redshift uncertainty
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        damping_function: np.ndarray
            Damping function due to GCsp redshift uncertainty
        """
        sigma_z = parameters['sigmaz']
        sigma_r = 299792.458 * sigma_z / \
            self.background_fiducial.hubble_parameter(parameters['z'],
                                                      units='km/s/Mpc')
        return np.exp(-k**2 * mu**2 * sigma_r**2)

    def _Pk2d_tot(self, k: np.ndarray, mu: np.ndarray,
                  parameters: dict) -> np.ndarray:
        r"""Total 2D power spectrum (including RSD, systematics, and noise)
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_tot: np.ndarray
            Total 2D power spectrum (including RSD, systematics, and noise)
        """
        return (self._Pk2d_rsd(k, mu, parameters) *
                self._damping_function(k, mu, parameters) *
                (1.0 - parameters['fout'])**2 +
                self._Pk2d_noise(k, mu, parameters))

    def power_multipoles(self, k: np.ndarray, parameters: dict,
                         ells: Optional[np.ndarray] = None,
                         use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        self.update(**parameters)
        params = parameters.copy()
        if params['As']<1e-7: params['As'] *= 1e9
        if 'H0' in params.keys(): params['h'] = params.pop('H0') / 100.0
        if 'omch2' in params.keys(): params['wc'] = params.pop('omch2')
        if 'ombh2' in params.keys(): params['wb'] = params.pop('ombh2')
        ells = self._ensure_array(ells) if ells else np.array([0,2,4])
        AP_factor = (self._q_AP_tr(parameters['z'])**2 *
                     self._q_AP_lo(parameters['z']) if use_AP else 1.0)
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / \
            AP_factor
        multipoles = {}
        for i,ell in enumerate(ells):
            multipoles[f'ell{ell}'] = \
                integrate.simps(self._Pk2d_tot(self._k_AP(k, self.mu_grid,
                                                          parameters['z'],
                                                          use_AP=use_AP),
                                               self._mu_AP(self.mu_grid,
                                                           parameters['z'],
                                                           use_AP=use_AP),
                                               params) *
                                legendre(ell)(self.mu_grid),
                                self.mu_grid, axis=1)
            multipoles[f'ell{ell}'] *= prefactors[i]
        return multipoles


class LegendreMultipolesComet(LegendreMultipoles):

    def __init__(self, NLmodel: Optional[str] = 'EFT',
                 linear_perturbations: Optional[LinearPerturbations] = None,
                 background_fiducial: Optional[Background] = None):
        r"""Class constructor
        Parameters
        ----------
        NLcode: str
            Non-linear code used to compute the power spectrum
        NLmodel: str
            Non-linear model (only EFT supported for now)
        linear_perturbations: LinearPerturbations
            Linear perturbations
        background_fiducial: Background
            Fiducial background (for AP corrections)
        """
        super().__init__(NLmodel=NLmodel,
                         linear_perturbations=linear_perturbations,
                         background_fiducial=background_fiducial)

        from comet import comet
        self.comet_inst = comet(model=NLmodel, use_Mpc=True,
                                bias_basis='AssBauGre')

    ######### FOR TESTING #########
    def set_fiducial_cosmology_comet(self, parameters: dict):
        self.comet_inst.define_fiducial_cosmology(params_fid=parameters)

    def power_multipoles_comet(self, k: np.ndarray, parameters: dict,
                               q_tr_lo: Optional[list] = None) -> dict:
        #self.comet_inst.define_fiducial_cosmology(params_fid=parameters)
        params = parameters.copy()
        if params['As']<1e-7: params['As'] *= 1e9
        return self.comet_inst.Pell(k=k, params=params, ell=[0,2,4],
                                    de_model='lambda', q_tr_lo=q_tr_lo)
    ###############################

    def _Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray,
                  parameters: dict) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields
        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        Pk2d_rsd: np.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        return self.comet_inst.Pk2d(k=k, mu=mu, params=parameters)
