"""Implementation of baryon correction of the matter power spectrum from FlamingoBaryonResponseEmulator on top of CAMB."""
# General imports
import numpy as np
from typing import Tuple, Optional
# Cosmology imports
from cloelib.cosmology.cosmology import Background
try:
    import camb  # type: ignore
    from camb import model  # type: ignore
except ImportError as e:
    raise ImportError("camb could not be imported.") from e
try:
    import FlamingoBaryonResponseEmulator as fre
except ImportError:
    raise ImportError("FlamingoBaryonResponseEmulator could not be imported.")
"""

## Notes:

- Adapted from FlamingoBaryonResponseEmulator/flamingo_response_emulator.py
- Adapted from cloelib/cloelib/cosmology/camb_cosmology.py

"""

class CAMBNonLinearFLAMINGOPerturbations:
    """A wrapper for CAMB nonlinear perturbation calculations with FLAMINGO correction for baryonic feedback"""

    def __init__(self, background: Background, redshifts: np.ndarray,
                  fgas_sigma: float, Mstar_sigma: float, jet_fraction: float, nonlinear_model: Optional[str] = None,) -> None:                 
        try:
            self.flamingo_emulator =  fre.FlamingoBaryonResponseEmulator()
        except ImportError:
            raise ImportError("FlamingoBaryonResponseEmulator could not be initialized.")                                     
        """
        Initialize the CAMBNonLinearPerturbations with FlamingoBaryonRespnseEmulator instance.

        Args:
        fgas_sigma: float
            The offset in numbers of sigma (of the X-ray data) the gas fraction
            in groups and clusters should be from the data used in the calibration
            of the FLAMINGO model. The emulator was trained between -8 and +2
            for a jet fraction of 0% and between -4 and 0 for a jet fraction of 100%.

        Mstar_sigma: float
            The offset in numbers of sigma (of the data) the stellar mass function
            should be from the data used in the calibration of the FLAMINGO model.
            The emulator was trained between -1 and 0.

        jet_fraction: float
            The fraction of the AGN energy released in the form of collimated jets
            (between 0 and 1). The original simulations exist only as purely thermal
            AGN (i.e. with jet = 0) and with purely collimated jets (i.e. with jet = 1).           
        linear_perturbations (LinearPerturbations): An instance of the LinearPerturbations class.
            redshifts (np.ndarray): Array of redshifts for the calculations.
        nonlinear_model (Optional[str]): The nonlinear model to use (e.g., "takahashi").
                Defaults to None, which uses the CAMB default model.
        """
        self.background = background
        self.kmax = 500
        self.z = redshifts
        self.fgas = fgas_sigma
        self.Mstar = Mstar_sigma
        self.jet = jet_fraction

        # Configure CAMB parameters for nonlinear calculations
        self.background.interface_args['CAMBparams'].NonLinear = model.NonLinear_both

        # Avoid unnecessary computations
        self.background.interface_args['CAMBparams'].WantCls = False
        self.background.interface_args['CAMBparams'].DoLensing = False
        self.background.interface_args['CAMBparams'].Want_CMB = False
        self.background.interface_args['CAMBparams'].Want_CMB_lensing = False
        self.background.interface_args['CAMBparams'].Want_cl_2D_array = False
        self.background.interface_args['CAMBparams'].WantTransfer = True
        
        if nonlinear_model:
            self.background.interface_args['CAMBparams'].NonLinearModel.set_params(halofit_version=nonlinear_model)

        self.background.interface_args['CAMBparams'].set_matter_power(redshifts=redshifts, kmax=self.kmax)

        # Compute nonlinear perturbations
        self.results = camb.get_results(self.background.interface_args['CAMBparams'])

        self.k, _, self.Pk = self.results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False)
        
    def matter_power_spectrum(self, zs, ks, hubble_units=False,
                              k_hunit=False) -> np.ndarray:
        r"""Compute the nonlinear matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber         

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk: numpy.ndarray
            Nonlinear matter power spectrum at the specified scale
            and redshift
        """
        pk_values = self.results.get_matter_power_interpolator(
            nonlinear=True, extrap_kmax=self.kmax,
            hubble_units=hubble_units, k_hunit=k_hunit,
            var1='delta_tot', var2='delta_tot').P(zs, ks) 
        flamingo_correction = self.baryonic_suppression(zs, ks,k_hunit=k_hunit)    
        return pk_values*flamingo_correction        

    def growth_rate(self) -> np.ndarray:
        """
        Calculate growth rate.

        Returns:
            np.ndarray: growth rate.
        """
        f_z = self.results.get_fsigma8()/self.results.get_sigma8()
        # Reversing array because camb re-sorts redshifts when power spectrum is computed
        return f_z[::-1]

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        np.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        D_z_k = np.sqrt(self.matter_power_spectrum(zs, ks) / \
                        self.matter_power_spectrum(0.0, ks))
        return D_z_k
        
    def baryonic_suppression(self, zs: np.array, ks: np.array, k_hunit=False) -> np.ndarray:
        """
        Return the predicted baryonic response for a set of comoving modes, redshift, and galaxy formation model (three parameters).

        Parameters
        ----------
        zs: np.array
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 3.
        

        ks: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of h Mpc^{-1}.
            
        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False            

        Returns
        -------
        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        Raises
        ------
        ValueError
            When the input redshift is not in the range [0, 3].

        """        
        response = np.ones((len(zs),len(ks)))
        if k_hunit:      
            for i in range(len(zs)): #FLAMINGO emulator only takes one redshift per call
            	response[i,:] = self.flamingo_emulator.predict(ks, zs[i], self.fgas, self.Mstar, self.jet)     
            return response	     
        else:
            for i in range(len(zs)): #FLAMINGO emulator only takes one redshift per call
                response[i,:] = self.flamingo_emulator.predict(ks*self.background.H0/100, zs[i], self.fgas, self.Mstar, self.jet) 
            return response
            
    def baryonic_suppression_with_variance(self, zs: np.array, ks: np.array, float, k_hunit=False) -> tuple[np.array, np.array]:
        """
        Return the predicted baryonic response as well as the variance around the prediction for a set of comoving modes, redshift, and galaxy formation model (three parameters).

        Parameters
        ----------
        zs: np.array
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 3.

        ks: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of Mpc^{-1}.
            
        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False            

        Returns
        -------
        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        baryon_ratio_variance: np.array
            The estimated variance of the baryonic response from the emulator
            at the modes k specified in the input.

        Raises
        ------
        ValueError
            When the input redshift is not in the range [0, 3].

        """ 
        response = np.ones((len(zs),len(ks)))             
        if k_hunit:      
            for i in range(len(zs)): #FLAMINGO emulator only takes one redshift per call
            	response[i,:] = self.flamingo_emulator.predict_with_variance(ks, zs[i], self.fgas, self.Mstar, self.jet)     
            return response	     
        else:
            for i in range(len(zs)): #FLAMINGO emulator only takes one redshift per call
                response[i,:] = self.flamingo_emulator.predict_with_variance(ks*self.background.H0/100, zs[i], self.fgas, self.Mstar, self.jet) 
            return response        

