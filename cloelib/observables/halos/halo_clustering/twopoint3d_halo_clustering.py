# import jax.numpy as np
import numpy as np

from cloelib.observables.halos.halo_clustering.halo_clustering_base import (
    HaloClusteringBase,
)


class TwoPoint3DHaloClustering(HaloClusteringBase):
    def power_spectrum_monopole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
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
        photoz_halo_corr = self.photoz_rsd_halo_correction(z, k, z_obs_scatter, b_eff)

        # dark matter power spectrum (z, k)
        pk = self.halo_model_properties.matter_power_spectrum_cb(z, k)

        # check if z_obs_scatter has more dimensions
        ndim_z_obs_scatter = len(np.array(z_obs_scatter).shape)
        if ndim_z_obs_scatter > 1:
            # if it does, add them to pk
            extra_axes = tuple(range(2, ndim_z_obs_scatter + 1))
            pk = np.expand_dims(pk, axis=extra_axes)

        # corrected power specrum (z, k, ...)
        pk_halo = photoz_halo_corr * pk

        return pk_halo

    def power_spectrum_quadrupole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        """Compute the halo power-spectrum quadrupole."""
        correction = self.photoz_rsd_halo_quadrupole_correction(
            z, k, z_obs_scatter, b_eff
        )
        pk = self.halo_model_properties.matter_power_spectrum_cb(z, k)

        ndim_z_obs_scatter = np.asarray(z_obs_scatter).ndim
        if ndim_z_obs_scatter > 1:
            pk = np.expand_dims(pk, axis=tuple(range(2, ndim_z_obs_scatter + 1)))

        return correction * pk

    def power_spectrum_hexadecapole_RSD_corrected(self, z, k, z_obs_scatter, b_eff):
        """Compute the halo power-spectrum hexadecapole."""
        correction = self.photoz_rsd_halo_hexadecapole_correction(
            z, k, z_obs_scatter, b_eff
        )
        pk = self.halo_model_properties.matter_power_spectrum_cb(z, k)

        ndim_z_obs_scatter = np.asarray(z_obs_scatter).ndim
        if ndim_z_obs_scatter > 1:
            pk = np.expand_dims(pk, axis=tuple(range(2, ndim_z_obs_scatter + 1)))

        return correction * pk

    def power_spectrum_RSD_amplitude(self, z, k, z_obs_scatter, b_eff, mu):
        """Compute the square-root halo power amplitude at fixed mu."""
        correction = self.photoz_rsd_halo_amplitude(z, k, z_obs_scatter, b_eff, mu)
        pk = self.halo_model_properties.matter_power_spectrum_cb(z, k)

        ndim_z_obs_scatter = np.asarray(z_obs_scatter).ndim
        if ndim_z_obs_scatter > 1:
            pk = np.expand_dims(pk, axis=tuple(range(2, ndim_z_obs_scatter + 1)))

        return correction * np.sqrt(pk)
