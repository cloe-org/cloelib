from cloelib.cosmology.cosmology import Perturbations
import numpy as np
from scipy.special import eval_legendre, spherical_jn
from scipy.integrate import simpson as simps


class HaloCovariance:
    def __init__(
            self,
            perturbations: Perturbations,
            k: np.ndarray,
            area: float,
            nbins_zob: int
        ):
        self.background = perturbations.background

        self.area = area
        self.k = k
        self.L = 20

        self.rint = np.zeros((nbins_zob, len(self.k), self.L + 1))

    def Kl_coeff(self):
        """
        Coefficients of the spherical harmonics expansion of the angular part of the window function

        Parameters
        ----------
        L: int
           Maximum number at which to evaluate the coefficients

        Returns
        -------
        KL: numpy.ndarray
            Coefficients up to L multipole
        """

        ell = np.linspace(0, self.L, self.L + 1, dtype=int)

        theta = np.arccos(1 - (self.area * (np.pi / 180.0) ** 2.0) / (2 * np.pi))

        KL = (
            np.sqrt(np.pi / (2.0 * ell + 1.0))
            * (
                eval_legendre(ell - 1, np.cos(theta))
                - eval_legendre(ell + 1, np.cos(theta))
            )
            / (2.0 * np.pi * (1 - np.cos(theta)))
        )

        KL[0] = 1 / (2.0 * np.sqrt(np.pi))

        return KL

    
    def cov_window(self, iz, zarr_iz, KL):
        """
        Computes the window function between redshifts bins

        Parameters
        ----------
        iz: int
            Index of the redshift bins at which to evaluate the window function
        zarr_iz: numpy.ndarray
             Array of redshifts (integration variable) between zbins[iz] and zbins[iz+1]
        KL: numpy.ndarray
            Spherical harmonic expansion coefficients

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i and j are two redshift bin and k are the wavenumbers
        """

        rvec = self.background.comoving_distance(zarr_iz) * (self.background.H0/100.)  # Mpc h^{-1}

        Vz = (rvec[-1] ** 3 - rvec[0] ** 3) / 3  # Mpc^3 h^{-3}

        kr = self.k[:, np.newaxis] * rvec

        self.rint[iz] = (
            1
            / Vz
            * simps(
                rvec**2.0
                * np.array(
                    [spherical_jn(l, kr, derivative=False) for l in range(self.L + 1)]
                ),
                x=rvec,
                axis=-1,
            ).T
        )
        return (4 * np.pi) * np.sum(
            self.rint[iz,:, :] * self.rint[: (iz + 1), :, :] * KL[:] ** 2, axis=-1
        )


