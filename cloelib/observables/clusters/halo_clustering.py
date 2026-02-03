# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import erf, spherical_jn

from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.selection_function import SelectionFunction

from ...auxiliary import units


class HaloClustering:
    def __init__(
        self,
        perturbations: Perturbations,
        perturbations_fid: Perturbations,
        selectionfunction: SelectionFunction,
        k: np.ndarray = np.geomspace(1e-4, 10, 500),
        nonu: bool = False,
    ):

        self.background = perturbations.background
        self.background_fid = perturbations_fid.background
        self.selectionfunction = selectionfunction
        self.nonu = nonu

        # wavelength array (integration variable)
        self.k = k

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
            self._Omega_m = self.background.Omega_m_cb
        else:
            self._Omega_m = self.background.Omega_m

    def WF_ra(self, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
                W[i,j,k] where i is the redshift bin, j is the radial bin and k are the wavenumbers
        spherical shell volume: numpy.ndarray
                V[i,j] where i is the redshift bin and j is the radial bin
        """

        r = r[np.newaxis, :, np.newaxis]
        k = self.k[np.newaxis, np.newaxis, :]

        r_z = (
            self.APcorr_func(z)[:, np.newaxis, np.newaxis] * r
        )  # AP correction (adds a redshift dependence)

        r3_TH_filter = (
            r_z**3
            * 3.0
            * (np.sin(k * r_z) - k * r_z * np.cos(k * r_z))
            / (k * r_z) ** 3.0
        )

        W_rad = (r3_TH_filter[:, 1:, :] - r3_TH_filter[:, :-1, :]) / (
            r_z[:, 1:, :] ** 3 - r_z[:, :-1, :] ** 3
        )

        V_rad = 4.0 * np.pi / 3.0 * ((r_z[:, 1:, 0]) ** 3 - (r_z[:, :-1, 0]) ** 3)

        return W_rad, V_rad

    # cosmo correction (isotropic AP)
    def APcorr_func(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the correction that accounts for the wrong cosmology assumed in the measurement of the 2ptCF
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
        Dv = (
            (1 + z) ** 2
            * self.background.angular_diameter_distance(z) ** 2
            * units.SPEED_OF_LIGHT
            * z
            / self.background.hubble_parameter(z)
        ) ** (1 / 3.0)

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = (
            (1 + z) ** 2
            * self.background_fid.angular_diameter_distance(z) ** 2
            * units.SPEED_OF_LIGHT
            * z
            / self.background_fid.hubble_parameter(z)
        ) ** (1 / 3.0)

        return (Dv / self.background.rdrag) * (self.background_fid.rdrag / Dv_fid)

    # IR resummation of the bao wiggles in the Pk
    def Pk_IR_func(self, Pk: np.ndarray) -> np.ndarray:
        """
        Infrared resummation (first order approx) to correct non-linear damping of bao wiggles

        Parameters
        ----------
        Pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3

        Returns
        -------
        Pk_IR: np.ndarray
           Matter power spectrum with corrected bao wiggles in (Mpc/h)^3

        """

        ns = self.background.ns
        h = self.background.h
        Obh2 = self.background.Omega_b(0.0) * h**2
        Omh2 = self._Omega_m(0.0) * h**2
        Tcmb = 2.73

        k = self.k
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

    def photoz_rsd_correction(
        self, z: np.ndarray, lambda_obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the correction that accounts for photo-z uncertainty and RSD (Kaiser effect)

        Parameters
        ----------
        z:  np.ndarray
            redshift
        lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        corr0, corr1, corr2: np.ndarray, np.ndarray, np.ndarray
            Correction terms to the power spectrum monopole
        """

        # growth rate
        f_gr = (self._Omega_m(z) ** 0.55)[:, np.newaxis]

        ks = self.k * (
            self.selectionfunction.scatter_z_obs(lambda_obs, z)
            * (units.SPEED_OF_LIGHT * 1e-3)
            / self.background.hubble_parameter(z)
            * (self.background.H0 / 100)
        ).reshape(len(z), 1)

        erf_ks = erf(ks)

        corr0 = np.sqrt(np.pi) / (2 * ks) * erf_ks
        corr1 = f_gr / ks**3 * (np.sqrt(np.pi) / 2 * erf_ks - ks * np.exp(-(ks**2)))
        corr2 = (
            f_gr**2
            / ks**5
            * (
                3 * np.sqrt(np.pi) / 8 * erf_ks
                - ks / 4 * (2 * ks**2 + 3) * np.exp(-(ks**2))
            )
        )

        # correct for numerical inaccuracy
        # note: for jax, use corr1 = corr1.at[idx].set(2 / 3.0)
        idx = erf_ks < 0.02
        corr1[idx] = 2 / 3.0
        corr2[idx] = 1 / 5.0

        return corr0, corr1, corr2
