# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn

from cloelib.auxiliary import units
from cloelib.cosmology.cosmology import Background
from cloelib.observables.clusters.auxiliary import (
    isotropic_volume_distance,
    photoz_rsd_correction,
    tophat_window,
)
from cloelib.observables.clusters.matter_statistics import MatterStatistics


class HaloClusteringCore:
    def __init__(
        self,
        matter_statistics: MatterStatistics,
        background_fid: Background,
        nonu: bool = False,
    ):

        self.matter_statistics = matter_statistics
        self.background_fid = background_fid
        self.nonu = nonu

    @property
    def nonu(self):
        r"""
        Includes or not neutrinos on matter density and matter power spectrum.
        """
        return self.__nonu

    @nonu.setter
    def nonu(self, value):
        """Set nonu"""
        if not isinstance(value, bool):
            raise ValueError(f"value for nonu must be boolean, used {value}")
        self.__nonu = value
        if self.nonu:
            self._Omega_m = self.matter_statistics.background.Omega_cb
        else:
            self._Omega_m = self.matter_statistics.background.Omega_m

    def radial_shell_window_and_volume(
        self, z: np.ndarray, k: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the window function and the volume of the spherical shells as a function of the radial separation

        Parameters
        ----------
        z: np.ndarray
           Redshift at which apply the geometrical correction (Alcock-Paczynski effect)
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
            shell_window, shape (z, r, k)
        spherical shell volume: numpy.ndarray
            shell_volume, shape (z, r)
        """

        r_z = (
            self.alcock_paczynski_correction_factor(z)[:, np.newaxis, np.newaxis]
            * r[np.newaxis, :, np.newaxis]
        )  # AP correction (adds a redshift dependence)
        k_r_z = r_z * k[np.newaxis, np.newaxis, :]

        shell_window = np.diff(r_z**3 * tophat_window(k_r_z), axis=1) / (
            np.diff(r_z**3, axis=1)
        )

        shell_volume = 4.0 * np.pi / 3.0 * np.diff(r_z[:, :, 0] ** 3, axis=1)

        return shell_window, shell_volume

    # cosmo correction (isotropic AP)
    def alcock_paczynski_correction_factor(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the Alcock-Paczynski correction factor for isotropic clustering measurements.
        See https://arxiv.org/pdf/1511.00012.pdf (Sect. 4.3.1) for details.

        Parameters
        ----------
        z: np.ndarray
           Redshift

        Returns
        -------
        AP_corr: np.ndarray
           Volume distance over drag scale (sound horizon scale at recombination) over the same quantity at fiducial cosmology

        """

        # units don't matter here, they cancel out

        z[z == 0] = 1e-5

        # isotropic volume distance
        Dv = isotropic_volume_distance(
            z,
            self.matter_statistics.angular_diameter_distance(z),
            self.matter_statistics.background.hubble_parameter(z),
        )

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = isotropic_volume_distance(
            z,
            self.background_fid.angular_diameter_distance(z),
            self.background_fid.hubble_parameter(z),
        )

        return (Dv / Dv_fid) * (
            self.background_fid.rdrag / self.matter_statistics.background.rdrag
        )

    def photoz_rsd_halo_correction(self, z, k, z_obs_scatter, b_eff):
        """Compute the correction that accounts for photo-z uncertainty
        and RSD (Kaiser effect) for halos.
        From `(Kaiser (1987)) <(https://doi.org/10.1093/mnras/227.1.1>`_.

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
        if z_obs_scatter.shape != b_eff.shape:
            raise ValueError(
                f"Shape of z_obs_scatter {z_obs_scatter.shape} must be"
                f" the same as b_eff {b_eff.shape}"
            )
        # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
        # rsd corrections (z, k, ...)
        photoz_corr0, photoz_corr1, photoz_corr2 = photoz_rsd_correction(
            self.matter_statistics.background, z, k, z_obs_scatter, self.nonu
        )

        # dark matter power spectrum (z, k)
        pk = self.matter_statistics.matter_power_spectrum(z, k)

        # halo correction (z, k, ...)
        b_eff_reshaped = b_eff[:, np.newaxis]  # add k axis in position 1
        photoz_halo_corr = (
            photoz_corr0 * b_eff_reshaped**2
            + photoz_corr1 * b_eff_reshaped
            + photoz_corr2
        )

        return photoz_halo_corr

    # IR resummation of the bao wiggles in the Pk
    # not in use currently
    def Pk_IR_func(self, k: np.array, Pk: np.ndarray) -> np.ndarray:
        """
        Infrared resummation (first order approx) to correct non-linear damping of bao wiggles.
        Approach of `(Eisenstein & Hu (1998)) <(https://doi.org/10.1086/305424>`_.

        Parameters
        ----------
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}
        Pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3

        Returns
        -------
        Pk_IR: np.ndarray
           Matter power spectrum with corrected bao wiggles in (Mpc/h)^3

        """

        ns = self.matter_statistics.background.ns
        h = self.matter_statistics.background.h
        Obh2 = self.matter_statistics.background.Omega_b(0.0) * h**2
        Omh2 = self._Omega_m(0.0) * h**2
        Tcmb = 2.73

        k *= h  #  1/Mpc
        s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1.0 + 10.0 * (Obh2) ** 0.75)
        Gamma = Omh2 / h
        AG = (
            1.0
            - 0.328 * np.log(431.0 * Omh2) * Obh2 / Omh2
            + 0.38 * np.log(22.3 * Omh2) * (Obh2 / Omh2) ** 2
        )
        Gamma = Gamma * (AG + (1.0 - AG) / (1.0 + (0.43 * k * s) ** 4))
        Theta = Tcmb / 2.7
        q = k * Theta**2 / Gamma / h
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T0 = L0 / (L0 + C0 * q * q)
        T0 /= T0[0]
        P_EH = k**ns * T0**2
        P_EH = (P_EH[:, None] * (Pk[:, 0] / P_EH[0])).T
        k /= h  # h/Mpc again

        lamb = 0.25
        kS = 0.2
        lOsc = 102.707
        qlog = np.log10(k)
        dqlog = qlog[1] - qlog[0]

        # Gaussian filtering for Pnw and Wiggle-smooth split
        Pnw = (
            P_EH
            * np.array(
                [
                    dqlog
                    / np.sqrt(2.0 * np.pi * lamb**2)
                    * (
                        np.sum(
                            np.e
                            ** (
                                -0.5
                                * (klog - qlog[(abs(qlog - klog) < 4.0 * lamb)]) ** 2
                                / lamb**2
                            )
                            * Pk[:, (abs(qlog - klog) < 4.0 * lamb)]
                            / P_EH[:, ((abs(qlog - klog) < 4.0 * lamb))],
                            axis=1,
                        )
                    )
                    for klog in qlog
                ]
            ).T
        )

        Pw = Pk - Pnw

        # Sigma2 as integral up to 0.2
        icut = k <= kS
        kcut = k[icut]
        Pnwcut = Pnw[:, icut]
        kosc = 1.0 / lOsc
        norm = 1.0 / (6.0 * np.pi**2)
        Sigma2 = norm * simps(
            Pnwcut
            * (1.0 - spherical_jn(0, kcut / kosc) + 2.0 * spherical_jn(2, kcut / kosc)),
            x=kcut,
        )

        # comput final power spectrum
        Pk_IR = Pnw + np.e ** (-(k**2) * Sigma2[:, None]) * Pw

        return Pk_IR
