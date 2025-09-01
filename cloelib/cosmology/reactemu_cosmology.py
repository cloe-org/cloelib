"""Implementation of Nonlinear Perturbation in extended cosmologies with ReACT."""
# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra


# General imports
import numpy as np
from typing import Tuple, Optional
from copy import deepcopy
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import interpolate


# Import nonlinear MG emulator module 
import MGEmu as mgemu



class MGemuNonlinearBoost:
    """Class for nonlinear boost using MGEmu: Boost = PNL_MG/PNL_ΛCDM."""

    def __init__(self, background : Background,
                 linearperturbations: Perturbations, zs: np.ndarray, 
                 gravity_model: str = 'fr',
                  mgpars: dict = {}):
        """Initialize the MGemuNonlinearBoost class to compute modified gravity (MG) nonlinear boost from emulators based on the Halo Model Reaction (https://arxiv.org/abs/1812.05594).

        These emulators were created based on training data produced using ReACT (https://arxiv.org/abs/2005.12184).
        This class allows for f(R), DGP, IDE, and other parameterized gravity models 
        See https://github.com/nebblu/MGEmus/tree/main for more details. 

        Parameters
        ----------
        background : Background
            A cosmological background instance containing parameters such as
            Omega_b0, Omega_cdm0, H0, ns, mnu, w0, and wa.

        linearperturbations : Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the ΛCDM baseline.

        zs : np.ndarray
            Array of redshifts at which to compute the MG corrections.

        gravity_model : str, optional
            Name of the gravity model or dark energy model to use.
            Examples: 'fr', 'dgp', 'gamma', or 'musigma-de'.

        mgpars : dict
            Dictionary of the extended MG parameters.
            Examples: fR0 for f(R), omegarc for DGP, gamma for Linder models, 
            mu0/Sigma0.

        """
        self.background = background

        # Cosmology imports
        try:
            MG_emu = mgemu.MG_boost(model = gravity_model, verbose=False)
        except ImportError:
            raise ImportError("MGEmu could not be imported or initialised.")


        # Setup redshifts to compute boost at 
        redshift_max = 2.4 # hard-coded for now as ReACT emulators will have this by default.

        # Split z into in-range and out-of-range
        zvals = zs 
        z_mask = zvals <= redshift_max
        zvals_inrange = zvals[z_mask]

        # Define the dictionary to be fed to the emulator
        # Add parameters as necessary 
        self.params_mg_emu = {
            'Omega_m': self.background.Omega_cdm0 + self.background.Omega_b0 + self.background.mnu/93.14/(self.background.H0 / 100)**2,
            'Omega_b': self.background.Omega_b0,
            'As': self.background.As,
            'ns': self.background.ns,
            'H0': self.background.H0,
            'Omega_nu': self.background.mnu/93.14/(self.background.H0 / 100)**2,
            'fR0':  mgpars['fr0'] if 'fr0' in mgpars else 1e-10,  # f(R) parameter
            'omegarc':  mgpars['omega_rc'] if 'omega_rc' in mgpars else 0.,  # DGP parameter
            'gamma':  mgpars['gamma0'] if 'gamma0' in mgpars else 0.55,  # Linder growth index
            'mu0':  mgpars['mu0'] if 'mu0' in mgpars else 0.,  # mu parameter for phenomenological models
            'sigma0':  mgpars['sigma0'] if 'sigma0' in mgpars else 0.,  # Sigma parameter for phenomenological models
            'q1':  mgpars['q1'] if 'q1' in mgpars else 0.,  # screening parameter
        }


        # Explicit bounds (if we homogenise the emulators this is fine, but otherwise should be able to call the bounds via cosmopower somehow)
        # names of parameters in which emulators were trained
        # later adapt for different emulator-ranges
        mg_bounds = {
            'Omega_m': [0.2, 0.6],
            'Omega_b': [0.03, 0.07],   #(0.04044, 0.05686),
            'H0':      [58., 80.],     #(63.8, 73.1),
            'ns':      [0.93, 1.],     #(0.9432, 0.9862),
            'As':      [0.5e-9, 5e-9], #(1.9511e-9, 2.2669e-9),
            'Omega_nu': [0., 0.1576],  #(1e-11, 0.1576),
            'fR0':  [1e-10,1e-4], 
            'omegarc':  [0.0,100], 
            'gamma':  [0.,1.], 
            'mu0':  [-0.999, 2.], 
            'sigma0':  [-0.999, 2.], 
            'q1':  [-2.,2.]
        }

        self.check_ranges = True
        verbose = False
        if self.params_mg_emu['mu0']>2.*self.params_mg_emu['sigma0']+1.:
            self.check_ranges = False
        # Copy parameter dict and clip/tile for in-range z only
        params_inrange = {}
        params_inrange['z'] = zvals_inrange
        for key in self.params_mg_emu.keys():
            if np.prod(self.params_mg_emu[key] - np.array(mg_bounds[key])) > 0:
                self.check_ranges = False
                if verbose:
                    print('parameter out of range: ', key, self.params_mg_emu[key])
                raise ValueError("MGemu emulator out of range.")
            else:
                params_inrange[key] = np.tile(self.params_mg_emu[key],
                                                  len(zvals_inrange))

        # Compute only for z ≤ redshift_max
        # k_emu in h/Mpc
        k_emu, boost_inrange = MG_emu.get_nonlinear_boost(**params_inrange)

        # Change h/Mpc --> 1/Mpc
        k_emu *= self.background.h

        # Create the interpolator over the original grid
        boost_inrange_interp = interpolate.RectBivariateSpline(zvals_inrange, k_emu, boost_inrange, kx=1, ky=1)


        # Low k extrapolation 
        # constant extrapolation of the boost to low  k 
        kmin = k_emu[0]
        kmax = k_emu[-1]
        
        k_low_mask = linearperturbations.k < kmax 
        k_target = linearperturbations.k[k_low_mask]

        # Precompute interpolated values on the new k grid
        boost_resampled = np.zeros((len(zvals_inrange), len(k_target)))

        for i, z_val in enumerate(zvals_inrange):
            for j, k_val in enumerate(k_target):
                if k_val < kmin: 
                    boost_resampled[i, j] = boost_inrange_interp(z_val, kmin)[0]
                else:
                    boost_resampled[i, j] = boost_inrange_interp(z_val, k_val)[0]


        # High k extrapolation 
        # Handle high-k extrapolation with extend_spectra or constant
        # For simplicity, assume constant high-k for now:

        # 🚀 Combine with extrapolation in z if needed:
        k_out, z_out, boost_out = extend_spectra(
            k_target, zvals_inrange, boost_resampled,
            flag_range=True,
            option_wavenumber="power_law",   # Or "const" if preferred for high-k
            option_redshift="power_law",     # Or 'none' if no z extrapolation needed
            extrap_z=zvals,                  # Full target z grid
            option_cosmo="const"
        )


        #✅ Apply boost = 1 for z > redshift_max if it is lower than 1 (this is safest thing to do since the extend_spectra results in unphysical high-z extrapolations)
        z_max = zvals_inrange[-1]
        z_mask_high = z_out > z_max
        if np.any(z_mask_high):
            # Only set boost to 1 where it is less than 1
            boost_out[z_mask_high, :] = np.maximum(boost_out[z_mask_high, :], 1.0)
        

        # Build interpolator
        self.MGboost_interp = interpolate.RectBivariateSpline(z_out, k_out, boost_out, kx=1, ky=1)



    def mg_spectrum_boost(self, zs, ks) -> np.ndarray:
        """Compute the nonlinear matter power spectrum boost.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        MGboost_interp: numpy.ndarray
            Nonlinear matter power spectrum boost at the specified scale
            and redshift

        """
        return self.MGboost_interp(zs, ks)



class BoostedPerturbations:
    """Class for nonlinear perturbations using MGEmu, inheriting from Perturbations parent class."""

    def __init__(self, base_lin_perturbations, 
                 base_perturbations, boost_interp):
        """Apply the nonlinear boost to the ΛCDM nonlinear spectrum given in base_perturbations.

        This initializes the BoostedPerturbations class, which applies a nonlinear boost 
        to the ΛCDM nonlinear spectrum provided by the base_perturbations object. The boost 
        is determined by the boost_interp function or interpolator.

        Parameters
        ----------
        base_lin_perturbations : object
            An object representing the linear perturbations, which may include methods 
            like `sigma_lensing` for lensing calculations.

        base_perturbations : object
            An object with a `matter_power_spectrum(z, k)` method that provides the 
            nonlinear matter power spectrum for the ΛCDM model.

        boost_interp : callable
            A function or interpolator B(z, k) that returns the nonlinear boost 
            to be applied to the ΛCDM spectrum.
        """
        self.background = base_perturbations.background
        assert self.background.Omega_k0 == 0, 'Non flat geometries not supported'
        assert self.background.w0==-1.0 and self.background.wa==0.0, 'All emulators are trained for ΛCDM background'


        self.base_lin = base_lin_perturbations
        if hasattr(base_lin_perturbations, 'sigma_lensing') and callable(getattr(base_lin_perturbations, 'sigma_lensing')):
            self.sigma_lensing = base_lin_perturbations.sigma_lensing


        self.base = base_perturbations
        self.boost_interp = boost_interp

        # Optionally expose attributes like z and k if they exist
        self.k = getattr(base_perturbations, 'k', None)
        self.z = getattr(base_perturbations, 'z', None)

    def matter_power_spectrum(self, z, k):
        """Return boosted nonlinear matter power spectrum P(k, z).

        Parameters:
            z : float or np.ndarray
            k : float or np.ndarray

        Returns:
            If z and k are arrays:
                ndarray with shape (len(z), len(k))
            If z is scalar and k is array:
                ndarray with shape (len(k),)
            If z is array and k is scalar:
                ndarray with shape (len(z),)
            If both are scalars:
                float
        """
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        # Get the unboosted spectrum from the base model
        P_base = self.base.matter_power_spectrum(z, k)

        # If both z and k are arrays, we expect shape (len(z), len(k))
        if P_base.shape == (len(z), len(k)):
            B = self.boost_interp(z[:, None], k[None, :], grid=False)
            P_boosted = B * P_base
        else:
            B = self.boost_interp(z, k)
            P_boosted = B * P_base

        # Squeeze to reduce unnecessary dimensions for plotting
        return np.squeeze(P_boosted)
    
  
    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Calculate the growth factor for given redshifts and wavenumbers.

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
        #if hasattr(self.base, 'Pk_interp') and self.base.Pk_interp is not None:
        #    D_z_k = np.sqrt((self.boost_interp(zs, ks)*self.base.Pk_interp(zs, ks))/ (self.boost_interp(0, ks)*self.base.Pk_interp(0, ks)))
        #
        #return D_z_k
        return self.base_lin.growth_factor(zs, ks)

    #def growth_rate(self, zs, ks) -> np.ndarray:
    #    """
    #    Calculate the growth rate for given redshifts and wavenumbers.
    #
    #    Returns:
    #    --------
    #    np.ndarray
    #        The growth rate as a function of redshift and wavenumber.
    #    """
    #    
    #
    #    return self.base_lin.growth_rate(zs, ks)
