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
    import MGrowth as mgrowth
except ImportError:
    raise ImportError("MGrowth could not be imported or initialised.")

"""

## Notes:

- Adapted from ABC classes

"""

class MGrowthLinearPerturbations:
    def __init__(self, background : Background, 
                 base_linear_perturbations: Perturbations, 
                 #redshifts: np.ndarray, 
                 gravity_model: str, 
                 mgpars: dict):
        """
        Initializes the MGLinearPerturbations class to compute modified gravity (MG)
        linear growth and rescaled matter power spectrum.

        This class augments a standard LCDM linear perturbation object with
        modifications to the growth factor and power spectrum using the MGrowth
        package, enabling f(R), DGP, IDE, and other parameterized gravity models.

        Parameters
        ----------
        background : Background
            A cosmological background instance containing parameters such as
            Omega_m, h, w0, and wa.

        base_linear_perturbations: Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the LCDM baseline.

        zs : np.ndarray
            Array of redshifts at which to compute the MG corrections.

        gravity_model : str
            Name of the MGrowth-supported gravity model to use.
            Examples: 'w0wacdm', 'fr', 'dgp', 'ide', 'gamma', 'gammaz', 'musigma-de'.

        mgpars : dictionary of the extended parameters
            Primary MG parameter 
            Examples: fR0 for f(R), omegarc for DGP, gamma0/gamma1 for Linder models, mu0/Sigma0 or binned values for mu-Sigma, xi for IDE,
            scrrening parameters etc.



        Notes
        -----
        This class ensures:
        - Redshifts are reversed to match MGrowth's expected ascending scale factors.
        - An interpolator for the gravitaional potential modification is computed.
        - If the model is scale-dependent (like f(R)), the returned growth factor D(z, k)
          is reshaped and ordered to match the base perturbation object's (z, k) convention.
        - The LCDM growth used for rescaling is also computed using MGrowth to maintain
          internal consistency.

        """


        assert background.Omega_k0 == 0, 'Non flat geometries not supported'




        self.background = background
        self.base = base_linear_perturbations

        # is called later in kernels, has to match
        self.z = self.base.z



        # Sort scale factors and redshifts in ascending order (early to late times)
        # hardcoded
        self.z_sorted = np.linspace(0., 5., 256, endpoint=True)
        # Get scale factors 
        self.a = 1. / (1. + self.z_sorted)
        self.a_sorted = self.a[::-1]
        

        # hardcoded to save time for f(R)
        self.k = np.logspace(np.log10(self.base.k[0]), np.log10(self.base.k[-1]), 128, endpoint=True )# in 1/Mpc
        #self.k = self.base.k
        self.k_len = len(self.k)

        self.gravity_model = gravity_model.lower()
        self.mgpars = mgpars

        # Build background dict for MGrowth
        background ={
            'Omega_m': self.background.Omega_m0,
            'h' : self.background.h,
            'w0': getattr(self.background, 'w0', -1.0),
            'wa': getattr(self.background, 'wa', 0.0),
            'a_arr': self.a_sorted
            }


        
        # Available MGrowth models (see https://github.com/MariaTsedrik/MGrowth)
        mg_models = {
            'w0wacdm': mgrowth.w0waCDM,
            'ide': mgrowth.IDE,
            'fr': mgrowth.fR_HS,
            'dgp': mgrowth.nDGP,
            'gamma': mgrowth.Linder_gamma,
            'gammaz': mgrowth.Linder_gamma_a,
            'musigma-de': mgrowth.mu_a,
        }
        if self.gravity_model not in mg_models:
            raise ValueError(f"Unsupported gravity model '{self.gravity_model}'.")

        # If musigma-de: specify mu-interp as a fucntion of dark energy evolution
        # parameterised by w0wa
        self.a_interp = np.linspace(1e-3, 1., 128)
        if self.gravity_model == 'musigma-de':
            if 'mu0' in mgpars and 'sigma0' in mgpars:
                self.mu_interp = self._compute_mu_de_interp(mgpars['mu0'], background['Omega_m'], background['w0'], background['wa'])
                self.sigma_lensing = self._compute_sigma_de_interp(mgpars['sigma0'], background['Omega_m'], background['w0'], background['wa'])
            else:
                raise ValueError('Mu-Sigma parameters are not properly specified.')    
        # Instantiate MGrowth cosmology
        self.mg_cosmo = mg_models[self.gravity_model](background)

        # Assign self._compute_growth_generic once instead of an if-statement
        # later other mu_interpolators can be added (e.g., binned or scale-dependent mu)
        if self.gravity_model == 'musigma-de':
            self._compute_growth_generic = getattr(self, '_compute_growth_muinterp')
        else:
            self._compute_growth_generic = getattr(self, f'_compute_growth_{self.gravity_model}', None)
        if self._compute_growth_generic is None:
            raise ValueError(f"Growth computation for model '{self.gravity_model}' is not implemented.")

        # Compute MG and LCDM growth and assign linear growth parameters
        self._compute_growth(background)

    def _compute_mu_de_interp(self, mu0, omega0, w0, wa):
        omegaL = (1.-omega0) * self.a_interp**(-3.*(1.+w0+wa)) * np.exp(3.*(-1.+self.a_interp)*wa)
        omegaL0 = (1.-omega0) 
        E = np.sqrt(omega0/self.a_interp**3 + omegaL)
        mu_de = 1. + mu0*(omegaL/E**2)/omegaL0
        mu_interpolator = interpolate.interp1d(self.a_interp, mu_de, bounds_error=False,
                kind='cubic',
                fill_value=(mu_de[0], mu_de[-1])) 
        # mu as a function of a (scale-factor)
        return  mu_interpolator  
    
    def _compute_sigma_de_interp(self, sigma0, omega0, w0, wa):
        omegaL = (1.-omega0) * self.z_sorted**(3.*(1.+w0+wa)) * np.exp(3.*(-1.+1./(1.+self.z_sorted))*wa)
        omegaL0 = (1.-omega0) 
        E = np.sqrt(omega0*self.z_sorted**3 + omegaL)
        sigma_de = 1. + sigma0*(omegaL/E**2)/omegaL0
        sigma_interpolator = interpolate.interp1d(self.z_sorted, sigma_de, bounds_error=False,
                kind='cubic',
                fill_value=(sigma_de[0], sigma_de[-1])) 
        # sigma as a function of z (redshift)
        return  sigma_interpolator  
        #sigma_de_k = np.repeat(sigma_de[:, None], self.k_len, axis=1)
        #sigma_interpolator = interpolate.RectBivariateSpline(self.z_sorted, self.k, sigma_de_k, kx=1, ky=1)
        ## sigma as a function of z (redshift) and k
        #return  sigma_interpolator 


    def _compute_growth_w0wacdm(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters()
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    
    def _compute_growth_dgp(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters(omegarc=self.mgpars['omega_rc'])
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    
    def _compute_growth_fr(self):    
        # Full k-dependent growth for f(R)
        D_raw, f_raw = self.mg_cosmo.growth_parameters(k_arr=self.k * self.background.h, fR0=self.mgpars['fr0'])
        # Transpose to get the arrays in (z,k), i.e. same shape as input power spectrum  
        D_raw, f_raw = D_raw.T, f_raw.T
        return D_raw[::-1, :], f_raw[::-1, :]
    
    
    def _compute_growth_ide(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters(xi=self.mgpars['xi'])
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    
    def _compute_growth_gamma(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters(gamma=self.mgpars['gamma0'])
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    
    def _compute_growth_gammaz(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters(gamma0=self.mgpars['gamma0'], gamma1=self.mgpars['gamma1'])
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    
    def _compute_growth_muinterp(self):    
        D_raw, f_raw = self.mg_cosmo.growth_parameters(mu_interp=self.mu_interp)
        return np.repeat(D_raw[::-1, None], self.k_len, axis=1), np.repeat(f_raw[::-1, None], self.k_len, axis=1)  

    def _compute_growth(self, bg_dict):
        """Handles model-specific MGrowth and LCDM growth evaluation."""
        D_mg, f_mg = self._compute_growth_generic()

        # Get LCDM growth and construct array over k to be applied as normalisation
        lcdm_cosmo = mgrowth.w0waCDM(bg_dict)
        D_lcdm_raw, _ = lcdm_cosmo.growth_parameters()
        D_lcdm = D_lcdm_raw[::-1]


        # Interpolate D(z, k) and f(z, k) 
        self.dz_interp = interpolate.RectBivariateSpline(self.z_sorted, self.k, D_mg, kx=1, ky=1)
        self.dz_norm_dz0_interp = interpolate.RectBivariateSpline(self.z_sorted, self.k, D_mg/D_mg[0, :], kx=1, ky=1)
        self.fz_interp = interpolate.RectBivariateSpline(self.z_sorted, self.k, f_mg, kx=1, ky=1)
        self.dz_norm_lcdm_interp = interpolate.RectBivariateSpline(self.z_sorted, self.k, D_mg/D_lcdm[0, None], kx=1, ky=1)





    def growth_factor(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers,
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
        

        return self.dz_norm_dz0_interp(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        return self.fz_interp(zs, ks)

        
        

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
        ps_base = self.base.matter_power_spectrum(0., ks)
        return self.dz_norm_lcdm_interp(zs, ks)**2 * ps_base

