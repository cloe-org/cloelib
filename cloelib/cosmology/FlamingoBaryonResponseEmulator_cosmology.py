"""Implementation of baryon correction of the matter power spectrum from FlamingoBaryonResponseEmulator."""
# General imports
import numpy as np
# Cosmology imports
try:
    import FlamingoBaryonResponseEmulator as fre
except ImportError:
    raise ImportError("FlamingoBaryonResponseEmulator could not be imported.")
"""

## Notes:

- Adapted from FlamingoBaryonResponseEmulator/flamingo_response_emulator.py

"""

class FlamingoBaryonResponseCorrection:
    """Class for power spectrum suppresion prediction from FLAMINGO using FlamingoBaryonResponseEmulator."""
    
    def __init__(self):
        """Intialize the FlamingoBaryonRespnseEmulator instance.""" 
        try:
            self.flamingo_emulator =  fre.FlamingoBaryonResponseEmulator()
        except ImportError:
            raise ImportError("FlamingoBaryonResponseEmulator could not be initialized.")        
                
    def predict(self, z: float, k: np.array, fgas_sigma: float, Mstar_sigma: float, jet_fraction: float) -> np.ndarray:
        """
        Return the predicted baryonic response for a set of comoving modes, redshift, and galaxy formation model (three parameters).

        Parameters
        ----------
        z: float
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 3.
        

        k: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of h Mpc^{-1}.

        sigma_gas: float
            The offset in numbers of sigma (of the X-ray data) the gas fraction
            in groups and clusters should be from the data used in the calibration
            of the FLAMINGO model. The emulator was trained between -8 and +2
            for a jet fraction of 0% and between -4 and 0 for a jet fraction of 100%.

        sigma_star: float
            The offset in numbers of sigma (of the data) the stellar mass function
            should be from the data used in the calibration of the FLAMINGO model.
            The emulator was trained between -1 and 0.

        jet: float
            The fraction of the AGN energy released in the form of collimated jets
            (between 0 and 1). The original simulations exist only as purely thermal
            AGN (i.e. with jet = 0) and with purely collimated jets (i.e. with jet = 1).

        Returns
        -------
        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        Raises
        ------
        ValueError
            When the input redshift is not in the range [0, 2].

        """        
        return self.flamingo_emulator.predict(k, z, fgas_sigma, Mstar_sigma, jet_fraction) 

    def predict_with_variance(self, z: float, k: np.array, fgas_sigma: float,  Mstar_sigma: float, jet_fraction: float) -> tuple[np.array, np.array]:
        """
        Return the predicted baryonic response as well as the variance around the prediction for a set of comoving modes, redshift, and galaxy formation model (three parameters).

        Parameters
        ----------
        z: float
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 3.

        k: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of h Mpc^{-1}.

        sigma_gas: float
            The offset in numbers of sigma (of the X-ray data) the gas fraction
            in groups and clusters should be from the data used in the calibration
            of the FLAMINGO model. The emulator was trained between -8 and +2
            for a jet fraction of 0% and between -4 and 0 for a jet fraction of 100%.

        sigma_star: float
            The offset in numbers of sigma (of the data) the stellar mass function
            should be from the data used in the calibration of the FLAMINGO model.
            The emulator was trained between -1 and 0.

        jet: float
            The fraction of the AGN energy released in the form of collimated jets
            (between 0 and 1). The original simulations exist only as purely thermal
            AGN (i.e. with jet = 0) and with purely collimated jets (i.e. with jet = 1).

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
            When the input redshift is not in the range [0, 2].

        """      
        return self.flamingo_emulator.predict_with_variance(k, z, fgas_sigma, Mstar_sigma, jet_fraction)         


