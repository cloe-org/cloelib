# cloelib imports
from cloelib.observables.tracer import Tracer
from cloelib.observables.photo import PositionsTracer
from cloelib.observables.photo import ShearTracer
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
import interpax
import jax.numpy as np
import jax

"""

## Notes:

- Two point asbtract class to compute two point functions
- Note, change for T vartype

"""

@jax.jit
def Cl_integration(WT1, WT2, Pkl, H, chi2):
    """
    Performs the integration to compute the angular power spectrum Cl.

    The integration is done using the unnormalized trapezoidal rule,
    utilizing the window functions, power spectrum, Hubble parameter,
    and comoving distance squared.

    Parameters:
    - WT1 (jax.numpy.ndarray): Window function for the first tracer.
    - WT2 (jax.numpy.ndarray): Window function for the second tracer.
    - Pkl (jax.numpy.ndarray): Matter power spectrum interpolated on Limber grid.
    - H (jax.numpy.ndarray): Hubble parameter evaluated at redshifts.
    - chi2 (jax.numpy.ndarray): Square of comoving distances at redshifts.

    Returns:
    - jax.numpy.ndarray: Angular power spectrum Cl with shape (len(ells), len(ells), len(ells)).
    """
    return np.einsum('iz,jz,lz,z,z->lij', WT1, WT2, Pkl, 1/H, 1/chi2)

@jax.jit
def Pkl_interp(k_l, z_l, ks, zs, Pk):
    """
    Interpolates the matter power spectrum on a Limber grid.

    Utilizes interpax's 2D interpolation with Akima method to handle
    non-uniform grids in logarithmic space. Extrapolation is enabled
    for values outside the given grid.

    Parameters:
    - k_l (jax.numpy.ndarray): Wavenumbers corresponding to (ells + 0.5) / chi.
    - z_l (jax.numpy.ndarray): Redshift grid for Limber integration.
    - ks (jax.numpy.ndarray): Original wavenumber grid of the matter power spectrum.
    - zs (jax.numpy.ndarray): Original redshift grid of the matter power spectrum.
    - Pk (jax.numpy.ndarray): Matter power spectrum values on (ks, zs) grid.

    Returns:
    - jax.numpy.ndarray: Interpolated power spectrum on the Limber grid.
    """
    return 10**interpax.interp2d(jax.numpy.log10(k_l), z_l, jax.numpy.log10(ks),  zs,
                                 jax.numpy.log10(Pk), method="akima", extrap=True)

Pkl_interp_vmap = jax.jit(jax.vmap(Pkl_interp, in_axes=(0, None, None, None, None)))


class AngularTwoPoint:
    def __init__(self, tracer1 : Tracer, tracer2 : Tracer):
        """
        Initializes the AngularTwoPoint object.

        Checks if the tracers are compatible and sets the two tracers
        as instance attributes.

        Parameters:
        - tracer1 (Tracer): The first tracer for the two-point function.
        - tracer2 (Tracer): The second tracer for the two-point function.
        """
        self.tracer1 = tracer1
        self.tracer2 = tracer2

    def _matter_power_spectrum_limber_grid(self, z_l, ks, zs, ells) -> jax.numpy.ndarray:
        """
        Prepares the matter power spectrum grid for Limber approximation.

        It calculates the k values on the Limber grid using the comoving
        distances and multipoles, then interpolates the matter power
        spectrum accordingly.

        Parameters:
        - z_l (jax.numpy.ndarray): Redshift grid for Limber integration.
        - ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.
        - zs (jax.numpy.ndarray): Redshift grid of the matter power spectrum.
        - ells (jax.numpy.ndarray): Multipole moments for angular power spectrum.

        Returns:
        - jax.numpy.ndarray: Interpolated matter power spectrum on the Limber grid.
        """
        chi = self.tracer1.perturbations.background.comoving_distance(z_l)
        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        Pk = self.tracer1.perturbations.matter_power_spectrum(zs, ks)
        Pkl = Pkl_interp_vmap(k_lz, z_l, ks, zs, Pk.T)
        return Pkl

    def get_Cl(self, ells, nl, ks)  -> jax.numpy.ndarray:
        """
        Computes the angular power spectrum Cl using Limber approximation.

        Combines the window functions of the tracers, interpolated matter power
        spectrum, Hubble parameter, and comoving distances to calculate the
        two-point angular statistics.

        Parameters:
        - ells (jax.numpy.ndarray): Multipole moments for the angular power spectrum.
        - nl (jax.numpy.ndarray): Noise power spectrum (not used yet, reserved for future use).
        - ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.

        Returns:
        - jax.numpy.ndarray: Angular power spectrum Cl for the given multipoles.
        """
        c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
        zs_calc = self.tracer1.z
        dz = self.tracer1.z[1]-self.tracer1.z[0]
        H = self.tracer1.perturbations.background.hubble_parameter(zs_calc, units = "km/s/Mpc")
        chi = self.tracer1.perturbations.background.comoving_distance(zs_calc)
        chi2 = chi**2
        WT1 = self.tracer1.get_window(zs_calc)
        WT2 = self.tracer2.get_window(zs_calc)
        Pkl = self._matter_power_spectrum_limber_grid(zs_calc, ks, self.tracer1.perturbations.z, ells)
        # Added the prefactor here as this is where we have access to ells.
        # There may be a more efficient way to do the multiplication
        prefactor = \
            (np.sqrt((ells + 2.0) * (ells + 1.0) * ells * (ells - 1.0)) /
             (ells + 0.5) ** 2)
        # Did it this way to avoid an if statement, but would be good to know how necessary this is
        prefactor_cell = ((prefactor * self.tracer1.prefact_toggle + 1 - self.tracer1.prefact_toggle) *
                          (prefactor * self.tracer2.prefact_toggle + 1 - self.tracer2.prefact_toggle))

        return c_0*Cl_integration(WT1, WT2, Pkl, H, chi2)*dz*prefactor_cell[:, None, None]

    def get_pseudo_Cl(self, nl, ks, mixing_matrix, n_ells_int=50)  -> jax.numpy.ndarray:
        """
        Computes the angular power spectrum Cl using Limber approximation 
        convolved with the mixing matrices.

        Combines the window functions of the tracers, interpolated matter power
        spectrum, Hubble parameter, and comoving distances to calculate the
        two-point angular statistics.

        Parameters:
        - nl (jax.numpy.ndarray): Noise power spectrum (not used yet, reserved for future use).
        - ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.
        - ks (numpy.ndarray): Mixing matrices in the euclidlib internal format.
        - n_ells_int (int): number of multiples to calculate.

        Returns:
        - jax.numpy.ndarray: Pseudo angular power spectrum Cl for the given multipoles.
        """
        ellmax = mixing_matrix[('POS', 'POS', 1, 1)].ell[-1]
        ells_calc = np.geomspace(1,ellmax+1,n_ells_int)
        C_ell_calc = self.get_Cl(ells_calc, nl, ks)
        C_ell_base = interpax.interp1d(np.arange(ellmax + 1),ells_calc,C_ell_calc,extrap=True)
        n_bin = C_ell_base.shape[2]

        C_ell_out = {}
        if (type(self.tracer1) == PositionsTracer) and (type(self.tracer2) == PositionsTracer):
            for i in range(1, n_bin+1):
                for j in range(i, n_bin+1):
                    C_ell_out[('POS','POS',i,j)] = mixing_matrix[('POS','POS',i,j)].array @ C_ell_base[:,i-1,j-1]

        elif (type(self.tracer1) == PositionsTracer) and (type(self.tracer2) == ShearTracer):
            for i in range(1, n_bin+1):
                for j in range(i, n_bin+1):
                    C_ell_out[('POS','SHE',i,j)] = mixing_matrix[('POS','SHE',i,j)].array @ C_ell_base[:,i-1,j-1]
                    C_ell_out[('POS','SHE',j,i)] = mixing_matrix[('POS','SHE',j,i)].array @ C_ell_base[:,j-1,i-1]

        elif (type(self.tracer1) == ShearTracer) and (type(self.tracer2) == PositionsTracer):
            for i in range(1, n_bin+1):
                for j in range(i, n_bin+1):
                    C_ell_out[('POS','SHE',j,i)] = mixing_matrix[('POS','SHE',j,i)].array @ C_ell_base[:,i-1,j-1]
                    C_ell_out[('POS','SHE',i,j)] = mixing_matrix[('POS','SHE',i,j)].array @ C_ell_base[:,j-1,i-1]

        elif (type(self.tracer1) == ShearTracer) and (type(self.tracer2) == ShearTracer):
            for i in range(1, n_bin+1):
                for j in range(i, n_bin+1):
                    C_ell_out[('SHE','SHE',i,j)] = np.stack([
                        mixing_matrix[('SHE','SHE',i,j)].array[0] @ C_ell_base[:,i-1,j-1],
                        mixing_matrix[('SHE','SHE',i,j)].array[1] @ C_ell_base[:,i-1,j-1]
                            ])
        return C_ell_out
