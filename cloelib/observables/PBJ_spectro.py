"""Interface of Legendre Multipoles with PBJ."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.cosmology.cosmology import Perturbations

# General imports
from typing import Protocol, Union, TypeVar, Optional
import numpy as np  # type: ignore
from copy import deepcopy

try:
    from pbjcosmo import PBJtheory
    pbj_obj = PBJtheory()
except:
    raise ImportError("PBJ could not be imported or initialised.")


class PBJSpectroPower:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PBJ."""

    def __init__(self, 
                 linear_perturbations: Perturbations,
                 nuisance_parameters: dict):
        r"""Class constructor.
    
        ### This docstring does not correspond to the function

        Args: 
          linear_perturbations (Perturbations): Perturbations object containing cosmology, linear power spectrum,
            redshift and growth functions
          nuisance_parameters (dict): Dictionary containing bias and counterterm parameters
          redshift (float): Redshift at which to evaluate $P(k,\mu)$
        """
        self.linear_perturbations = linear_perturbations
        self.background = linear_perturbations.background
        self.parameters = nuisance_parameters
        self.redshift = linear_perturbations.z

        self.cosmo = {'h':    self.background.h,
                      'Och2': self.background.Omega_cdm0 * self.background.h**2,
                      'Obh2': self.background.Omega_b0   * self.background.h**2,
                      'As':   self.background.As,
                      'ns':   self.background.ns,
                      'Mnu':  self.background.mnu,
                      'w0':   self.background.w0,
                      'wa':   self.background.wa,
                      'Tcmb': 2.7255}

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        ### This docstring does not correspond to the function
        
        Args: 
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight
          parameters (dict): Ensemble of cosmological and nuisance parameters

        Returns: 
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        plinear = self.linear_perturbations.matter_power_spectrum(
            0., pbj_obj.kL, hubble_units=False, k_hunit=False)
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units='1/Mpc')

        pkmu = pbj_obj.P_kmu_2D(
            self.redshift[0], True, kgrid=k, mu=mu,
            f=self.linear_perturbations.growth_rate(),
            D=self.linear_perturbations.growth_factor(self.redshift, 0.05)[0],
            cosmo=self.cosmo, IRres=True, **self.parameters)

        return pkmu
        
    def Pk2d_X_rsd(self, k: np.ndarray, mu: np.ndarray, X: str) -> np.ndarray:
        r"""2D power spectrum for the specific diagram X.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight
          X (str): Identifier of loop diagram

        Returns: 
          PX2d_rsd (np.ndarray): 2D power spectrum of term X
        """
        plinear = self.linear_perturbations.matter_power_spectrum(
            0., pbj_obj.kL, hubble_units=False, k_hunit=False)
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units='1/Mpc')

        pkmu_marg_dict = pbj_obj.P_kmu_2D_marg(
            self.redshift[0], True, kgrid=k, mu=mu,
            f=self.linear_perturbations.growth_rate(),
            D=self.linear_perturbations.growth_factor(self.redshift, 0.05)[0],
            cosmo=self.cosmo, IRres=True, **self.parameters)
        # note: make this work with X=list/array of strings?

        return pkmu_marg_dict[X]
