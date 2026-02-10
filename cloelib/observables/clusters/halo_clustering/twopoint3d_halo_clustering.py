# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn

from cloelib.auxiliary import units
from cloelib.cosmology.cosmology import Background
from cloelib.observables.clusters.auxiliary import (
    photoz_rsd_correction,
)
from cloelib.observables.clusters.halo_clustering.halo_clustering_core import (
    HaloClusteringCore,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class TwoPoint3DHaloClustering:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        background_fid: Background,
        nonu: bool = False,
    ):

        self.core = HaloClusteringCore(matter_statistics, background_fid, nonu)

    def power_spectrum_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        """Computes Pk with RSD correction.

        Parameters
        ----------
        z: np.ndarray
           Redshift
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}
        pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3
           Shape (z.size, k.size)
        z_obs_scatter: float, numpy.ndarray
            Observed redshift scatter. If array, first dimension must be z.
        b_eff: np.ndarray
            Effective bias, must have same shape as z_obs_scatter.

        Returns
        -------
        pk_halo : numpy.ndarray
            Power spectrum averaged on redshift and richnesses bins (with IR-resummation),
            Shape (z.size, k.size, other dimensions of z_obs_scatter)
        """
        # correct halos power specrum for photo-z uncertainties and RSD (eqs. 80-83)
        # halo rsd corrections (z, k, ...)
        photoz_halo_corr = self.core.photoz_rsd_halo_correction(
            z, k, z_obs_scatter, b_eff
        )

        # dark matter power spectrum (z, k)
        pk = self.core.matter_statistics.matter_power_spectrum(z, k)

        # check if z_obs_scatter has more dimensions
        ndim_z_obs_scatter = len(np.array(z_obs_scatter).shape)
        if ndim_z_obs_scatter > 1:
            # if it does, add them to pk
            extra_axes = tuple(range(2, ndim_z_obs_scatter + 1))
            pk = np.expand_dims(pk, axis=extra_axes)

        # corrected power specrum (z, k, ...)
        pk_halo = photoz_halo_corr * pk

        return pk_halo
