# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra


# General imports
import numpy as np
from scipy.interpolate import RectBivariateSpline
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cosmopower as cp

REFERENCE_SPECTRUM_FILE = (
    "/home/bbose/cosmopower/data/npz/stable_new/"
    "stable_reference_spectrum_350k_b1_v1.npz"
)


class mochiBoostLinearPerturbations:
    """
    Emulator for the linear matter power spectrum, trained on mochi CLASS, using Cosmopower
    based on the stable basis parametrisation.
    """

    def __init__(self,
                background: Background,
                base_perturbations: Perturbations,
                zs: np.ndarray,
                cp_file: str,
                stable_MG_dict: dict,
                high_z_policy: str = "taper",
                z_decay: float = 0.75):
        """
        Initialize the emulator with a given cosmological background and redshift array.

        Parameters
        ----------
        background : Background
            Background cosmology object, providing all necessary cosmological parameters.
        base_perturbations : Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the LCDM baseline.
        zs : np.ndarray
            Array of redshift values for which the power spectrum should be computed.
            Must be sorted in ascending order.
        cp_file : str
            Path to the file containing the trained Cosmopower emulator for linear perturbations.
        stable_MG_dict : dict
            Dictionary containing the MG stable basis parameters.
        high_z_policy : str
            Extrapolation policy for redshifts above the emulator range.
            One of "power_law", "freeze", "one", "taper".
        z_decay : float
            Decay rate with which the boost returns to LCDM (boost=1) above the
            emulator max redshift. Only used by the "taper" policy.
        Raises
        ------
        AssertionError
            If the geometry is not flat (Omega_k0 != 0).
        ValueError
            If any parameter lies outside the bounds supported by the emulator,
            if `zs` is not sorted ascending, or if `high_z_policy` is unknown.
        """
        if high_z_policy not in {"power_law", "freeze", "one", "taper"}:
            raise ValueError(
                f"Unknown high_z_policy: {high_z_policy}. Choose from "
                '"power_law", "freeze", "one", "taper".'
            )

        zs = np.asarray(zs, dtype=float)
        if np.any(np.diff(zs) <= 0):
            raise ValueError("zs must be sorted in strictly ascending order.")

        self.background = background
        self.base_perturbations = base_perturbations
        self.z_decay = float(z_decay)
        self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        self.k_emu = self.cp_LIN.modes.copy()
        self.k_emu *= background.h

        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        self.params = {
            "s": stable_MG_dict["s"],
            "a0": stable_MG_dict["a0"],
            "a1": stable_MG_dict["a1"],
            "M02": stable_MG_dict["M02"],
            "w0": background.w0,
            "wa": background.wa,
        }
        emu_bounds = {
            "w0": np.array([-2.0, 0.5]),
            "wa": np.array([-2.0, 1.0]),
            "s": np.array([-0.5, 0.5]),
            "a0": np.array([0.0, 5.0]),
            "a1": np.array([-5.0, 0.0]),
            "M02": np.array([0.5, 1.5]),
            "z": np.array([0.0, 5.0]),
        }


        # CURRENTLY TRAINED WITH FIXED COSMOLOGY (PLANCK 2018):
        # fixed_Omega_m = 0.315
        # fixed_Omega_b = 0.049
        # fixed_H0 = 67.4
        # fixed_ns = 0.9649
        # fixed_As = 2.1e-9
        # fixed N_mnu = 0
        # fixed mnu = 0

        # The emulator is only queried at the redshifts inside its training
        # range, so the MG parameters must be tiled to that same length.
        redshift_max = emu_bounds["z"][1]
        zvals_inrange = zs[zs <= redshift_max]

        for key in self.params.keys():
            if np.prod(self.params[key] - emu_bounds[key]) > 0:
                raise ValueError("Cosmopower emulator out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(zvals_inrange))

        self.params["z"] = zvals_inrange

        logratio_inrange = self.cp_LIN.ten_to_predictions_np(self.params)

        ref = np.load(REFERENCE_SPECTRUM_FILE)
        ref_spectrum = np.asarray(ref["reference_spectrum"], dtype=float)

        pk_emu = ref_spectrum[None, :] * logratio_inrange / background.h**3
        pk_base = np.asarray(
            self.base_perturbations.matter_power_spectrum(zvals_inrange, self.k_emu),
            dtype=float,
        )
        boost_inrange = pk_emu / pk_base

        # Create the interpolator over the original grid
        boost_inrange_interp = RectBivariateSpline(
            zvals_inrange, self.k_emu, boost_inrange, kx=1, ky=1
        )

        k_low_mask = base_perturbations.k < self.k_max
        k_target = base_perturbations.k[k_low_mask]

        # Precompute interpolated values on the new k grid
        boost_resampled = np.zeros((len(zvals_inrange), len(k_target)))
        vals_k = np.clip(k_target, self.k_min, None)

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
                extrap_z=zs,  # full requested z grid
                option_cosmo="const",
            )
        else:
            # extend only in k; keep z at the emulator's valid range
            # (some versions use "none", others "const" to disable z extrap)
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

            # 3) now fill z > redshift_max according to the policy
            z_out = np.asarray(zs)
            boost_out = np.empty((len(z_out), len(k_out)))

            # zvals_inrange was selected with the same condition, so this mask
            # necessarily has as many True entries as boost_kext has rows.
            idx_in = z_out <= redshift_max
            idx_hi = ~idx_in
            boost_out[idx_in, :] = boost_kext

            last = boost_kext[-1, :]  # slice at z=redshift_max

            if high_z_policy == "one":
                boost_out[idx_hi, :] = 1.0

            elif high_z_policy == "freeze":
                boost_out[idx_hi, :] = last[None, :]

            elif high_z_policy == "taper":
                dz = (z_out[idx_hi] - redshift_max)[:, None]
                boost_out[idx_hi, :] = 1.0 + (last - 1.0)[None, :] * np.exp(
                    -dz / self.z_decay
                )

        # ✅ Apply boost = 1 for z > redshift_max if it is lower than 1 (this is safest thing to do since the extend_spectra results in unphysical high-z extrapolations)
        enforce_floor_highz = False
        if enforce_floor_highz:
            hi = z_out > redshift_max  # z_max = 2 in current setup
            if np.any(hi):
                boost_out[hi, :] = np.maximum(boost_out[hi, :], 1.0)

        # Build interpolator
        self.MGboost_interp = RectBivariateSpline(z_out, k_out, boost_out, kx=1, ky=1)

        # Expose the grids the boost is defined on, as the base perturbations do
        self.k = k_out
        self.z = z_out

    def matter_power_spectrum(self, z, k):
        """Return the boosted linear matter power spectrum.

        Parameters
        ----------
        z : float or np.ndarray
            Redshift value or array of redshifts.
        k : float or np.ndarray
            Wavenumber value or array of wavenumbers in 1/Mpc.

        Returns
        -------
        float or np.ndarray
            Boosted matter power spectrum. The output is squeezed so scalar
            inputs return a scalar, while array inputs return the corresponding
            one- or two-dimensional array.
        """
        z = np.atleast_1d(z)
        k = np.atleast_1d(k)

        # Get the unboosted spectrum from the base model
        P_base = self.base_perturbations.matter_power_spectrum(z, k)

        # If both z and k are arrays, we expect shape (len(z), len(k))
        if P_base.shape == (len(z), len(k)):
            B = self.MGboost_interp(z[:, None], k[None, :], grid=False)
            P_boosted = B * P_base
        else:
            B = self.MGboost_interp(z, k)
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
        # Pick a default large-scale k: the smallest *emulated* mode. Note that
        # self.k extends below it, but there the boost is 1 by construction of
        # the low-k extrapolation, which would wash out the growth entirely.
        if getattr(self, "k_min", None) is not None:
            k_lin = float(self.k_min)  # safest large-scale mode available
        else:
            k_lin = 2e-2  # [1/Mpc] fallback default; adjust if your units differ

        # Evaluate boost at (z, k_lin) and (0, k_lin)
        # RectBivariateSpline returns 2D arrays; slice to 1D
        Bz = self.MGboost_interp(z, np.array([k_lin]))  # shape (nz, 1)
        Bz = Bz[:, 0]
        B0 = float(self.MGboost_interp(np.array([0.0]), np.array([k_lin]))[0, 0])

        B0 = max(B0, 1e-300)  # safety
        D_lin = np.sqrt(Bz / B0)

        return np.squeeze(D_lin)

    def growth_rate(self, ks=None, zs=None) -> np.ndarray:
        r"""Calculate the modified-gravity growth rate.

        The MG growth rate is obtained from the GR (base) one by adding the
        logarithmic derivative of the boost with respect to the scale factor.
        Written directly in terms of redshift (using
        :math:`d\ln a = -d\ln(1+z)`), this is

        .. math::
            f_{\rm MG}(z) = f_{\rm GR}(z)
                          - \frac{1+z}{2}\frac{d\ln B(z)}{dz},

        where :math:`B` is the emulated boost on the linear matter power
        spectrum.

        Parameters
        ----------
        ks : array_like or None
            Wavenumbers in 1/Mpc at which the (scale-dependent) boost
            correction is evaluated. If None, the largest available emulated
            scale (`self.k_min`) is used, so that the result is a
            scale-independent growth rate.
        zs : array_like or None
            Redshifts. Defaults to the grid the boost is defined on
            (`self.z`), matching the convention of the other perturbation
            classes, whose `growth_rate()` returns an array over `self.z`.

        Returns
        -------
        np.ndarray
            If `ks` is None or scalar -> shape (len(zs),)
            If both are arrays        -> shape (len(zs), len(ks))
        """
        z = self.z if zs is None else np.atleast_1d(zs).astype(float)

        # GR growth rate of the baseline (LCDM) perturbations, on its own z
        # grid, resampled onto the requested redshifts.
        f_GR = self._base_growth_rate(z)

        if ks is None:
            k = np.array([float(self.k_min)])
            squeeze_k = True
        else:
            k = np.atleast_1d(ks).astype(float)
            squeeze_k = np.ndim(ks) == 0

        # B and dB/dz on the (z, k) grid from the boost spline. `dx=1` is the
        # derivative in the spline's first argument, i.e. in z.
        B = self.MGboost_interp(z, k)
        dBdz = self.MGboost_interp(z, k, dx=1)

        B = np.clip(B, 1e-300, None)

        # dlnB/dz = (dB/dz) / B
        dlnB_dz = dBdz / B

        f_MG = f_GR[:, None] - 0.5 * (1.0 + z)[:, None] * dlnB_dz

        if squeeze_k:
            f_MG = f_MG[:, 0]

        return np.squeeze(f_MG)

    def _base_growth_rate(self, z: np.ndarray) -> np.ndarray:
        """Return the base (GR) growth rate evaluated at the redshifts `z`.

        The baseline perturbation classes expose `growth_rate()` with slightly
        different signatures: some take no argument and return f on their own
        redshift grid (e.g. `CLASSLinearPerturbations`), others take
        (zs, ks) directly. Both are supported here.
        """
        try:
            f_base = np.asarray(self.base_perturbations.growth_rate(), dtype=float)
        except TypeError:
            # Signature expects the redshifts (and possibly wavenumbers).
            return np.atleast_1d(
                np.asarray(
                    self.base_perturbations.growth_rate(z, self.k_min), dtype=float
                )
            ).astype(float)

        z_base = np.atleast_1d(
            np.asarray(getattr(self.base_perturbations, "z", z), dtype=float)
        )
        f_base = np.atleast_1d(f_base)

        if f_base.shape == z.shape and np.allclose(z_base, z):
            return f_base

        # Interpolate onto the requested redshifts (z_base must be ascending).
        order = np.argsort(z_base)
        return np.interp(z, z_base[order], f_base[order])

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        B0 = float(self.MGboost_interp(np.array([0.0]), np.array([2e-2]))[0, 0])
        return float(self.base_perturbations.sigma8_0() * np.sqrt(B0))
