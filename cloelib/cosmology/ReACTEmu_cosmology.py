# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra


# General imports
import numpy as np
from typing import Tuple, Optional
from copy import deepcopy
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import interpolate


# Cosmology imports
try:
    import MGEmu as mgemu
except ImportError:
    raise ImportError("MGEmu could not be imported or initialised.")


# Naming conventions for parameters and models match MGrowth module (see https://github.com/MariaTsedrik/MGrowth)

class MGemuNonlinearBoost:
    def __init__(self, background : Background,
                 linearperturbations: Perturbations, 
                 zs: np.ndarray, 
                 gravity_model: str, 
                 mgpars: dict):
        

        """
        Initializes the MGemuNonlinearBoost class to compute modified gravity (MG)
        nonlinear boost from emulators based on the Halo Model Reaction (https://arxiv.org/abs/1812.05594)

        These emulators were created based on training data produced using ReACT (https://arxiv.org/abs/2005.12184)

        This class allows for f(R), DGP, IDE, and other parameterized gravity models 

        See https://github.com/nebblu/MGEmus/tree/main for more details 

        Parameters
        ----------
        background : Background
            A cosmological background instance containing parameters such as
            Omega_b0, Omega_cdm0, H0, ns, mnu, w0, and wa.

        linearperturbations : Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the LCDM baseline.

        zs : np.ndarray
            Array of redshifts at which to compute the MG corrections.

        gravity_model : str, optional
            Name of the gravity model or dark energy model to use.
            Examples: 'fr', 'dgp', 'ide', 'gamma', 'lcdm', etc.

        mgpars : dict
            Dictionary of the extended MG parameters.
            Examples: fR0 for f(R), omegarc for DGP, gamma0/gamma1 for Linder models, 
            mu0/Sigma0 or binned values for mu-Sigma, xi for IDE, etc.

        """

        self.background = background

        # Cosmology import 
        MG_emu = mgemu.MG_boost(model = gravity_model)

        # Setup redshifts to compute boost at 
        redshift_max = 2. # hard-coded for now as ReACT emulators will have this by default.

        # Split z into in-range and out-of-range
        zvals = zs 
        z_mask = zvals <= redshift_max
        zvals_inrange = zvals[z_mask]

        # Available MGemu models (see https://github.com/nebblu/MGEmus)
        # EDIT: if new model available, add tag here 
        allowed_models = {"fr", "dgp", "gamma"}


        if self.gravity_model not in allowed_models:
            raise ValueError(
                f"Unsupported gravity model '{self.gravity_model}'. "
                f"Choose one of: {', '.join(sorted(allowed_models))}"
            )



        # Define the dictionary to be fed to the emulator
        # Add parameters as necessary 
        # EDIT: if new model available, add parameters here 
        self.params_mg_emu = {
            'Omega_m': self.background.Omega_cdm0 + self.background.Omega_b0 + self.background.mnu/93.14/(self.background.H0 / 100)**2,
            'Omega_b': self.background.Omega_b0,
            'As': self.background.As,
            'ns': self.background.ns,
            'H0': self.background.H0,
            'Omega_nu': self.background.mnu/93.14/(self.background.H0 / 100)**2,
            'fR0':  mgpars['fr0'],
            'omegarc':  mgpars['omega_rc'],
            'gamma':  mgpars['gamma0'],
            'q1':  mgpars['q1'],
            'z': zvals
        }


        # Explicit bounds (if we homogenise the emulators this is fine, but otherwise should be able to call the bounds via cosmopower somehow)
        mg_bounds = {
            'Omega_m': (0.2899, 0.3392),
            'Omega_b': (0.04044, 0.05686),
            'H0':      (63.8, 73.1),
            'ns':      (0.9432, 0.9862),
            'As':      (1.9511e-9, 2.2669e-9),
            'Omega_nu': (1e-11, 0.1576),
            'fR0':  (1e-10,1e-4), 
            'omegarc':  (0.001,100), 
            'gamma':  (0.,1.), 
            'q1':  (-5.,5.),
            'z' : (0 , redshift_max) 
        }


        # Copy parameter dict and clip/tile for in-range z only
        params_inrange = {}
        for key, (lower, upper) in mg_bounds.items():
            if key == 'z':
                params_inrange['z'] = zvals_inrange
            else:
                val = self.params_mg_emu[key]
                val_clipped = np.clip(val, lower, upper)
                params_inrange[key] = np.tile(val_clipped, len(zvals_inrange))

        # Compute only for z ≤ redshift_max
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
        r"""Computes the nonlinear matter power spectrum boost.

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
    def __init__(self, base_perturbations, boost_interp):
        """
        Applies the nonlinear boost to the LCDM nonlinear spectrum given in base_perturbations 

        Parameters
        ----------
        base_perturbations : object
            An object with a `matter_power_spectrum(z, k)` method.
        
        boost_interp : callable
            A function or interpolator B(z, k) that returns the nonlinear boost.
        """
        self.base = base_perturbations
        self.boost_interp = boost_interp

        # Optionally expose attributes like z and k if they exist
        self.k = getattr(base_perturbations, 'k', None)
        self.z = getattr(base_perturbations, 'z', None)

    def matter_power_spectrum(self, z, k):
        """
        Returns boosted nonlinear matter power spectrum P(k, z)

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
