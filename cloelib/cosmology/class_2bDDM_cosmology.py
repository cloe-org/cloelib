"""Implementation of Background and Perturbation cosmology using CLASS."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
import numpy as np
import copy
from typing import Optional, Union, Sequence
import DMemu

# Cosmology imports
try:
    from classy import Class  # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e


import os
import sys
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# TODO : implement classes for Background and LinearPerturbations
# (needs interface with classy corresponding to my modified version of CLASS)

class tbDDMNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    c0 = SPEED_OF_LIGHT / 1000
    
    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        f_dcdm: float,
        epsilon: float, 
        Gamma: float
    ):
        """Initialize the tbDDMNonLinearPerturbations instance."""
        self.background = background
        self.z = redshifts
        self.kmax = 30

        self.f     = f_dcdm # fraction of cdm that decays
        self.Gamma = Gamma # decay rate in 1/Gyr
        self.vk    = epsilon*tbDDMNonLinearPerturbations.c0 # velocity fraction in km/s

        # Initialize NN emulator
        self.emul = DMemu.TBDemu()

        # Ensure CLASS is initialized with necessary parameters
        self.interface_args = copy.deepcopy(self.background.interface_args)
        self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
        self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
        self.interface_args["CLASSparams"]["k_per_decade_for_bao"] = 70
        self.interface_args["CLASSparams"]["k_per_decade_for_pk"] = 10
        self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
        self.interface_args["CLASSparams"]["nonlinear_min_k_max"] = 50
        self.interface_args["CLASSparams"]["non linear"] = "halofit" # note that the fitting formula is defined as a "boost" wrt halofit-LCDM
        self.results = Class()
        self.results.set(self.interface_args["CLASSparams"])
        self.results.compute()
        # GFA, I added this line in order to retrieve the wavenumber grid (in 1/Mpc) used by CLASS to compute Pk
        _, self.k, _ = self.results.get_pk_and_k_and_z(nonlinear=True, only_clustering_species = False, h_units=False)

    def boost_2bDDM(self, z, k) -> float:
        """Calculate the boost factor for the 2bDDM suppression, with the emulator by Bucko et al. (2307.03222)
        
        Parameters
        ----------
        z: float
           redshift
        k: float
           wavenumber

        Returns
        -------
        S_2bDDM: float
                boost factor for the 2bDDM suppression at given redshift and wavenumber
        """
        if (z > 2.35):
            return 1.0 # we shouldn't extrapolate the emulator beyond the training domain for z
                          # Extrapolation for kappa > 6 h/Mpc is done by adding a constant suppression continuously attached
                          # to the one provided by an emulator
        else:
            with SuppressOutput():
                S_2bDDM = self.emul.predict(np.asarray(k).reshape(-1),
                                            np.asarray(z).reshape(-1),
                                            self.f,
                                            self.vk,
                                            self.Gamma, 
                                            allow_z_extrapolation = False)
        
            return np.asarray(S_2bDDM).item()


    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the CLASS non-linear matter power spectrum.

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
            Non-linear matter power spectrum at the specified scale
            and redshift
        """
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
    
        self.Pk_nonlinear = np.array(
            [[self.results.pk(ki, zi)*self.boost_2bDDM(zi, ki) for ki in ks] for zi in zs] # we apply the boost factor for the 2bDDM suppression
        )
        # To match array convention of CAMB
        return self.Pk_nonlinear

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
        # NOTE: a priori here we should use the "linear" matter power spectrum for 2bDDM
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.zeros_like(zs), ks)
        )

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        arr = [self.results.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)
