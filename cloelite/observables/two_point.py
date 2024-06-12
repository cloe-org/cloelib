import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RectBivariateSpline
import jax

# cloelite imports
from cloelite.observables.tracer import Tracer

"""

## Notes:

- Two point asbtract class to compute two point functions

"""

class TwoPoint(ABC):
    def __init__(self, tracer1 : Tracer, tracer2 : Tracer):
        # The only common ingredients to all the tracers are
        # perturbations and cosmological background
        # (inherited from perturbations too)
        if type(tracer1.perturbations) != type(tracer2.perturbations):
            TypeError("The types of the perturbations of the two tracers is not compatible!")

        self.tracer1 = tracer1
        self.tracer2 = tracer2

class AngularTwoPoint(TwoPoint):
    def __init__(self, tracer1 : Tracer, tracer2 : Tracer):
        super().__init__(tracer1, tracer2)

    def get_Cl(self):
        #now hardcoded, later probably some hyper parameters to pass to the constructor
        nl = 100
        ells = np.logspace(1., np.log10(3000), nl)
        ks = np.logspace(-5, 2, 300)

        zs_calc = self.tracer1.z
        ks = np.logspace(-5, 2, 300)
        H = self.tracer1.background.comoving_distance(zs_calc)
        chi = self.tracer1.background.comoving_distance(zs_calc)


        chi2 = chi**2
        Pk = jax.vmap(self.tracer1.perturbations.nonlinear_matter_power_spectrum,
                      in_axes = (0, None))(ks, zs_calc)
        pmm_logspline = RectBivariateSpline(np.log10(ks), zs_calc, np.log10(Pk),
                                            kx = 3, ky = 3, s = 0)
        Pkl = np.zeros((nl, len(zs_calc)))
        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        for (z_idx, myz) in enumerate(zs_calc):
            Pkl[:, z_idx] = 10**pmm_logspline(np.log10(k_lz[:, z_idx]), myz)[:,0]

        WT1 = self.tracer1.get_window_positions(zs_calc)
        WT2 = self.tracer1.get_window_positions(zs_calc)
        result = np.einsum('iz,jz,lz,z,z->lij', WT1, WT2, Pkl, 1/H, 1/chi2)
        return result
