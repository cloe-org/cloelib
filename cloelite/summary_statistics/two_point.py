# cloelite imports
from cloelite.observables.tracer import Tracer
from cloelite.auxiliary.units import SPEED_OF_LIGHT

# General imports
import interpax
import jax.numpy as np
from scipy.interpolate import RectBivariateSpline
import jax

"""

## Notes:

- Two point asbtract class to compute two point functions

"""

@jax.jit
def Cl_integration(WT1, WT2, Pkl, H, chi2):
    # To be updated
    # still have to include weights, basically we are doing unnormalized trapz
    return np.einsum('iz,jz,lz,z,z->lij', WT1, WT2, Pkl, 1/H, 1/chi2)

@jax.jit
def Pkl_interp(k_l, z_l, ks, zs, Pk):
    return 10**interpax.interp2d(jax.numpy.log10(k_l), z_l, jax.numpy.log10(ks),  zs,
                                 jax.numpy.log10(Pk), method="akima", extrap=True)

Pkl_interp_vmap = jax.jit(jax.vmap(Pkl_interp, in_axes=(0, None, None, None, None)))


class AngularTwoPoint:
    def __init__(self, tracer1 : Tracer, tracer2 : Tracer):
        # include test to check if tracers are compatible
        self.tracer1 = tracer1
        self.tracer2 = tracer2

    def _matter_power_spectrum_limber_grid(self, z_l, ks, zs, ells) -> np.ndarray:
        """
        Prepares the matter power spectrum to calculate two-point angular
        statistics following Limber
        """
        chi = self.tracer1.perturbations.background.comoving_distance(zs)
        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        Pk = self.tracer1.perturbations.matter_power_spectrum()
        Pkl = Pkl_interp_vmap(k_lz, z_l, ks, zs, Pk.T)
        return Pkl

    def get_Cl(self, ells, nl, ks)  -> np.ndarray:

        c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
        zs_calc = self.tracer1.z
        dz = self.tracer1.z[1]-self.tracer1.z[0]
        H = self.tracer1.perturbations.background.hubble_parameter(zs_calc)
        chi = self.tracer1.perturbations.background.comoving_distance(zs_calc)
        chi2 = chi**2
        WT1 = self.tracer1.get_window(zs_calc)
        WT2 = self.tracer2.get_window(zs_calc)
        Pkl = self._matter_power_spectrum_limber_grid(zs_calc, ks, zs_calc, ells)

        return c_0*Cl_integration(WT1, WT2, Pkl, H, chi2)*dz

    def get_pseudo_Cl(self, ells, nl, ks, mixing_matrix)  -> np.ndarray:
        pass
