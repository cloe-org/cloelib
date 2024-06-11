import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import RectBivariateSpline

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
        #now hardcoded, later probably some hyper parameters to pass to the constructore
        ks = np.logspace(-5, 2, 300)
        zs = np.logspace(0., 6., 300)
        ell_range = np.logspace(1., 3., 50)

        zs_calc = np.linspace(0,6,3001)#horrible Marco!
        H = self.tracer1.background.comoving_distance(zs_calc)
        chi = self.tracer1.background.comoving_distance(zs_calc)
        ks_calc = np.outer(ell_range + 1, 1/chi)


        chi2 = chi**2
        Pk = self.tracer1.perturbations.nonlinear_matter_power_spectrum(ks, zs)
        pmm_logspline = RectBivariateSpline(zs, np.log10(ks), np.log10(Pk))
                                                        #kx=self.angular_config.limber_spline_kx,
                                                        #ky=self.angular_config.limber_spline_ky,
                                                        #s=self.angular_config.limber_spline_s)
        Pkl = np.zeros((50, 3001))
        k_lz = np.expand_dims((ell_range + 0.5), 1) / chi
        for (z_idx, myz) in enumerate(zs_calc):
            Pkl[:, z_idx] = pmm_logspline(myz, np.log10(k_lz[:, z_idx]))


        Pkl = np.zeros((50, 3001))
        for lidx in range(50):
            Pkl[lidx,:] = Pk_interp(ks_calc[lidx], chi)
        WT1 = self.tracer1.get_window_positions(zs_calc)
        WT2 = self.tracer1.get_window_positions(zs_calc)
        result = np.einsum('iz,jz,lz,z,z->l,i,j', WT1, WT2, Pk, 1/H, 1/chi2)
        return result
