"""
Tabulated nonlinear boost module.

This module provides a lightweight interface for using simulation-based
nonlinear matter power spectrum boosts B(k, z).  It reads a text file
containing a common k-grid and boost values for a set of snapshot
redshifts, constructs a 2D spline interpolator B(z, k), and exposes it in
the required format.
"""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra


# General imports
import numpy as np
from typing import Sequence
from scipy.interpolate import RectBivariateSpline


class TabulatedNonlinearBoost:
    """
    Nonlinear matter power spectrum boost from a tabulated file B(k; z).

    The boost file is expected to have:
      - column 1: k values (same k-grid used for all redshifts)
      - columns 2..N: boost values B(k; z_i) for each snapshot/redshift

    Parameters
    ----------
    background : Background
        A cosmological background instance containing parameters such as
        Omega_b0, Omega_cdm0, H0, ns, mnu, w0, and wa. Note w0 and wa are assumed to be LCDM values for some modified gravity models.

    linearperturbations : Perturbations
        A standard linear perturbation object (e.g. from CAMB) used as the LCDM baseline.

    zs : np.ndarray
        Array of redshifts at which to compute the MG corrections.

    boost_file : str
        Path to the text file containing [k, B(z1), B(z2), ..., B(zN)].
        Commented header lines starting with '#' are allowed.

    z_cols : Sequence[float]
        1D array or list of redshifts corresponding to columns 2..N in the
        boost file (same order). Length must match the number of boost columns.


    high_z_policy: string (default taper)
        Gives the high redshift extrapolation policy. Can choose from the following:
            options: "power_law", "freeze", "one", "taper"

    z_decay: float
        Gives the decay rate after emulator max redshift going back to LCDM (boost=1)

    Notes
    -----
    This class simply builds a RectBivariateSpline over (z, k) so that you can
    pass `self.boost_interp` into `BoostedPerturbations` in exactly the same
    way as the ReACT emulator-based version.
    """

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        zs: np.ndarray,
        boost_file: str,
        z_cols: Sequence[float],
        high_z_policy: str = "taper",
        z_decay: float = 0.75,
    ):
        # Load table: k, B(z1), B(z2), ...
        data = np.loadtxt(boost_file)

        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"Boost file '{boost_file}' must have at least 2 columns "
                "(k and one boost column)."
            )

        # k grid
        ktab = data[:, 0]
        boost_kz = data[:, 1:]  # shape (Nk, Nsnap)

        z_cols = np.asarray(z_cols, dtype=float)
        if boost_kz.shape[1] != len(z_cols):
            raise ValueError(
                f"Number of boost columns ({boost_kz.shape[1]}) does not match "
                f"length of z_cols ({len(z_cols)})."
            )

        # Ensure z is sorted ascending and reorder boost columns accordingly
        sort_idx = np.argsort(z_cols)
        z_sorted = z_cols[sort_idx]
        boost_zk = boost_kz[
            :, sort_idx
        ].T  # -> shape (Nz, Nk) as RectBivariateSpline expects

        zmin = z_sorted[0]
        zmax = z_sorted[-1]

        self.background = background
        # Change h/Mpc --> 1/Mpc
        ktab *= self.background.h

        # Build 2D spline B(z, k)
        # Note: assumes k and z arrays are strictly increasing
        boost_inrange_interp = RectBivariateSpline(z_sorted, ktab, boost_zk, kx=1, ky=1)

        # ---- choose redshift extrapolation policy ----
        # options: "power_law", "freeze", "one", "taper"
        # validate & store policy
        if high_z_policy not in {"power_law", "freeze", "one", "taper"}:
            raise ValueError("Invalid high_z_policy ...")

        self.z_decay = float(z_decay)

        # 1) Low k extrapolation
        # constant extrapolation of the boost to low  k
        kmin = ktab[0]
        kmax = ktab[-1]

        k_low_mask = linearperturbations.k < kmax
        k_target = linearperturbations.k[k_low_mask]

        zvals = zs
        zvals_inrange = zvals[(zvals >= zmin) & (zvals <= zmax)]

        # Precompute interpolated values on the new k grid
        boost_resampled = np.zeros((len(zvals_inrange), len(k_target)))
        vals_k = np.clip(k_target, kmin, None)

        for i, z_val in enumerate(zvals_inrange):
            boost_resampled[i, :] = boost_inrange_interp(z_val, vals_k)[0]

        # High k extrapolation
        # Handle high-k extrapolation with extend_spectra or constant
        # For simplicity, assume constant high-k for now:

        # 2) use extend_spectra for *k*; choose z-policy

        if high_z_policy == "power_law":
            # let extend_spectra also do z with its power-law
            k_out, z_out, boost_out = extend_spectra(
                k_target,
                zvals_inrange,
                boost_resampled,
                flag_range=True,
                option_wavenumber="power_law",
                option_redshift="power_law",
                extrap_z=zvals,  # full requested z grid
                option_cosmo="const",
            )

        else:
            # extend only in k; keep z at the grid values
            try:
                k_out, z_in, boost_kext = extend_spectra(
                    k_target,
                    zvals_inrange,
                    boost_resampled,
                    flag_range=True,
                    option_wavenumber="power_law",
                    option_redshift="none",
                    extrap_z=zvals_inrange,
                    option_cosmo="const",
                )
            except TypeError:
                k_out, z_in, boost_kext = extend_spectra(
                    k_target,
                    zvals_inrange,
                    boost_resampled,
                    flag_range=True,
                    option_wavenumber="power_law",
                    option_redshift="const",
                    extrap_z=zvals_inrange,
                    option_cosmo="const",
                )

            # 3) now fill z > zmax according to the policy
            z_out = np.asarray(zvals)
            boost_out = np.empty((len(z_out), len(k_out)))

            idx_in = (z_out >= zmin) & (z_out <= zmax)
            idx_hi = ~idx_in
            boost_out[idx_in, :] = boost_kext

            last = boost_kext[-1, :]  # slice at z=zmax

            if high_z_policy == "one":
                boost_out[idx_hi, :] = 1.0

            elif high_z_policy == "freeze":
                boost_out[idx_hi, :] = last[None, :]

            elif high_z_policy == "taper":
                dz = (z_out[idx_hi] - zmax)[:, None]
                boost_out[idx_hi, :] = 1.0 + (last - 1.0)[None, :] * np.exp(
                    -dz / self.z_decay
                )

            else:
                raise ValueError(f"Unknown high_z_policy: {high_z_policy}")

        # ✅ Apply boost = 1 for z > zmax if it is lower than 1 (this is safest thing to do since the extend_spectra results in unphysical high-z extrapolations)
        enforce_floor_highz = False
        if enforce_floor_highz:
            hi = z_out > zmax  # z_max = 2 in current setup
            if np.any(hi):
                boost_out[hi, :] = np.maximum(boost_out[hi, :], 1.0)

        # Build interpolator
        self.MGboost_interp = RectBivariateSpline(z_out, k_out, boost_out, kx=1, ky=1)

    def mg_spectrum_boost(self, zs, ks) -> np.ndarray:
        r"""Computes the nonlinear matter power spectrum boost.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        MGboost_interp: numpy.ndarray
            Nonlinear matter power spectrum boost at the specified scale
            and redshift

        """

        return self.MGboost_interp(zs, ks)


class TabulatedBoostedPerturbations:
    def __init__(self, base_lin_perturbations, base_perturbations, boost_interp):
        """
        Applies the nonlinear boost to the LCDM nonlinear spectrum given in base_perturbations

        Parameters
        ----------
        Parameters
        ----------
        base_lin_perturbations : object
            An object representing the linear perturbations, which may include methods
            like `sigma_lensing` for lensing calculations.

        base_perturbations : object
            An object with a `matter_power_spectrum(z, k)` method that provides the
            nonlinear matter power spectrum for the ΛCDM model.

        boost_interp : callable
            A function or interpolator B(z, k) that returns the nonlinear boost
            to be applied to the ΛCDM

        """

        self.background = base_perturbations.background
        assert self.background.Omega_k0 == 0, "Non flat geometries not supported"

        self.base = base_perturbations
        self.boost_interp = boost_interp

        self.base_lin = base_lin_perturbations
        if hasattr(base_lin_perturbations, "sigma_lensing") and callable(
            getattr(base_lin_perturbations, "sigma_lensing")
        ):
            self.sigma_lensing = base_lin_perturbations.sigma_lensing

        # Optionally expose attributes like z and k if they exist
        self.k = getattr(base_perturbations, "k", None)
        self.z = getattr(base_perturbations, "z", None)

    def matter_power_spectrum(self, z, k):
        """
        Returns boosted nonlinear matter power spectrum P(k, z)

        Parameters:
            z : float or np.ndarray
            k : float or np.ndarray

        Returns:
            If z and k are arrays:
                ndarray with shape (len(z), len(k))
            If z is scalar and k is array:
                ndarray with shape (len(k),)
            If z is array and k is scalar:
                ndarray with shape (len(z),)
            If both are scalars:
                float
        """
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        # Get the unboosted spectrum from the base model
        P_base = self.base.matter_power_spectrum(z, k)

        # If both z and k are arrays, we expect shape (len(z), len(k))
        if P_base.shape == (len(z), len(k)):
            B = self.boost_interp(z[:, None], k[None, :], grid=False)
            P_boosted = B * P_base
        else:
            B = self.boost_interp(z, k)
            P_boosted = B * P_base

        # Squeeze to reduce unnecessary dimensions for plotting
        return np.squeeze(P_boosted)

    def growth_factor(self, zs, ks=None):
        r"""Calculate the growth factor.

        If `ks` is provided:
            D(z,k) = sqrt( P(z,k) / P(0,k) )
        If `ks` is None (linear growth via boost on large scales):
            D(z)   = sqrt( B(z,k_lin) / B(0,k_lin) ), with a sensible default k_lin.

        Parameters
        ----------
        zs : array_like
            Redshifts.
        ks : array_like or None
            Wavenumbers. If None, returns linear-growth estimate from the boost at k_lin.

        Returns
        -------
        np.ndarray
            If ks is None or scalar -> shape (len(zs),)
            If z and k arrays       -> shape (len(zs), len(ks))
        """
        z = np.atleast_1d(zs).astype(float)

        # Case A: fully general growth via power spectra ratio
        if ks is not None:
            k = np.atleast_1d(ks).astype(float)
            P = self.matter_power_spectrum(z, k)  # shape (nz, nk) or (nk,) if z scalar
            P0 = self.matter_power_spectrum(np.array([0.0]), k)  # shape (1, nk)
            P0 = np.squeeze(P0)
            # avoid div-by-zero in pathological cases
            P0 = np.clip(P0, 1e-300, None)
            D = np.sqrt(P / P0)
            return np.squeeze(D)

        # Case B: linear-growth estimate from boost at large scales
        # pick a default large-scale k: prefer the smallest available k-grid if we have one
        if getattr(self, "k", None) is not None and len(self.k) > 0:
            k_lin = float(
                np.min(self.k)
            )  # typically the safest large-scale mode available
        else:
            k_lin = 2e-2  # [1/Mpc] fallback default; adjust if your units differ

        # Evaluate boost at (z, k_lin) and (0, k_lin)
        # RectBivariateSpline returns 2D arrays; slice to 1D
        Bz = self.boost_interp(z, np.array([k_lin]))  # shape (nz, 1)
        Bz = Bz[:, 0]
        B0 = float(self.boost_interp(np.array([0.0]), np.array([k_lin]))[0, 0])

        B0 = max(B0, 1e-300)  # safety
        D_lin = np.sqrt(Bz / B0)

        return np.squeeze(D_lin)

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        B0 = self.boost_interp(np.array([0.0]), np.array([2e-2]))  # shape (nz, 1)
        B0 = np.sqrt(B0[:, 0])
        return self.base.sigma8_0() * B0
