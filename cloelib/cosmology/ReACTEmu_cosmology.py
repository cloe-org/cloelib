# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.extrapolator import extend_spectra


# General imports
import numpy as np
from scipy.interpolate import RectBivariateSpline


# Cosmology imports
try:
    import MGEmu as mgemu
except ImportError:
    raise ImportError("MGEmu could not be imported or initialised.")


# Naming conventions for parameters and models match MGrowth module (see https://github.com/MariaTsedrik/MGrowth)

######### ADDING NEW EMULATORS #########
# If you are adding in a new model you need to do the following:
#  - Add emulator tag to MODEL_TO_BACKEND
#  - Include the cosmological and model priors to BOUNDS_BY_MODEL
#  - Include model parameters to params_mg_emu


class MGemuNonlinearBoost:
    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        zs: np.ndarray,
        gravity_model: str,
        mgpars: dict,
        high_z_policy: str = "taper",
        z_decay: float = 0.75,
    ):
        """
        Initializes the MGemuNonlinearBoost class to compute modified gravity (MG)
        nonlinear boost from emulators based on the Halo Model Reaction (https://arxiv.org/abs/1812.05594)

        These emulators were created based on training data produced using ReACT (https://arxiv.org/abs/2005.12184)

        This class allows for f(R), DGP, IDE and other parameterized gravity models

        See https://github.com/nebblu/MGEmus/tree/main for more details

        Parameters
        ----------
        background : Background
            A cosmological background instance containing parameters such as
            Omega_b0, Omega_cdm0, H0, ns, mnu, w0, and wa. Note w0 and wa are assumed to be LCDM values for some modified gravity models.

        linearperturbations : Perturbations
            A standard linear perturbation object (e.g. from CAMB) used as the LCDM baseline.

        zs : np.ndarray
            Array of redshifts at which to compute the MG corrections.

        gravity_model : str, optional
            Name of the gravity model or dark energy model to use.
            Examples: 'fr', 'dgp', 'ide', 'wCDM', 'w0waCDM', 'gamma', 'mu'  etc.

        mgpars : dict
            Dictionary of the extended MG parameters.
            Examples: fR0 for f(R), omegarc for DGP, gamma0/gamma1 for Linder models,
            mu0/Sigma0 or binned values for mu-Sigma, xi for IDE, etc.

        high_z_policy: string (default taper)
            Gives the high redshift extrapolation policy. Can choose from the following:
             options: "power_law", "freeze", "one", "taper"

        z_decay: float
            Gives the decay rate after emulator max redshift going back to LCDM (boost=1)

        """

        self.background = background
        self.gravity_model = gravity_model  # <-- set it

        # Available MGemu models (see https://github.com/nebblu/MGEmus)
        # EDIT: if new model available, add tag here

        # Map user-facing tag -> emulator backend tag
        MODEL_TO_BACKEND = {
            "fr": "fr",
            "dgp": "dgp",
            "gamma": "gamma",
            "mu": "mu",
            # DS family -> single backend "ds"
            "wCDM": "ds",
            "w0waCDM": "ds",
            "ide": "ds",
            "ds": "ds",
        }

        allowed_models = set(MODEL_TO_BACKEND.keys())
        if self.gravity_model not in allowed_models:
            raise ValueError(
                f"Unsupported gravity model '{self.gravity_model}'. "
                f"Choose one of: {', '.join(sorted(allowed_models))}"
            )

        # Load emulator for the mapped backend model
        backend_model = MODEL_TO_BACKEND[self.gravity_model]
        MG_emu = mgemu.MG_boost(model=backend_model)

        # -------------------------------
        # Per-model bounds (COSMO + MG)
        # If sampling, try to stick to these priors!
        # Hard-code your training boxes here.
        # If a key is missing for a model, it won't be clipped.
        # -------------------------------
        BOUNDS_BY_MODEL = {
            # --- f(R) ---
            "fr": {
                "Omega_m": (0.24, 0.35),
                "Omega_b": (0.04, 0.06),
                "H0": (63.0, 75.0),
                "ns": (0.9, 1.01),
                "As": (1.7e-9, 2.5e-9),
                "Omega_nu": (1e-11, 0.00317),
                "fR0": (1e-10, 1e-4),
                "z": (0.0, 2.0),
            },
            # --- DGP ---
            "dgp": {
                "Omega_m": (0.2899, 0.3392),
                "Omega_b": (0.04044, 0.05686),
                "H0": (62.9, 73.1),
                "ns": (0.9432, 0.9862),
                "As": (1.5e-9, 2.7e-9),
                "Omega_nu": (1e-11, 0.00317),
                "omegarc": (1e-3, 100.0),
                "z": (0.0, 2.4),
            },
            # --- Linder gamma ---
            "gamma": {
                "Omega_m": (0.2899, 0.3392),
                "Omega_b": (0.04044, 0.05686),
                "H0": (63.8, 73.1),
                "ns": (0.9432, 0.9862),
                "As": (1.9511e-9, 2.2669e-9),
                "Omega_nu": (1e-11, 0.00317),
                "gamma": (0.0, 1.0),
                "q1": (-5.0, 5.0),
                "z": (0.0, 2.4),
            },
            # --- mu(k;z) with screening  ---
            "mu": {
                "Omega_m": (0.24, 0.4),
                # "Omega_b": (0.049, 0.049), # Fixed for this model
                "H0": (60.0, 84.0),
                # "ns":      (0.9649, 0.9649), # Fixed for this model
                "As": (1.7e-9, 2.5e-9),
                # "Omega_nu": (0.0, 0.0), # Fixed for this model
                "w0": (-1.5, -0.5),
                "wa": (-0.5, 0.5),
                "mu0": (-0.999, 3.0),
                "c1": (-0.3333, 1.0),
                "lam": (0.0, 2.0),
                "q1": (-2.0, 2.0),
                "q2": (-2.0, 2.0),
                "q3": (-2.0, 2.0),
                "z": (0.0, 2.5),
            },
            # --- DS family backend ("ds") ---
            # Replace the placeholder COSMO/MG ranges below with the real training domain.
            "ds": {
                "Omega_m": (0.22, 0.37),
                "Omega_b": (0.03, 0.08),
                "H0": (63.0, 84.0),
                "ns": (0.8, 1.1),
                "As": (1.7e-9, 2.5e-9),
                "Omega_nu": (1e-11, 0.00317),
                "w0": (-1.3, -0.5),
                "wa": (-2.0, 0.5),
                "xi": (0.0, 150.0),
                "z": (0.0, 2.5),
            },
        }

        # pick bounds for the explicit user tag; fall back to backend if needed
        bounds = BOUNDS_BY_MODEL.get(
            self.gravity_model, BOUNDS_BY_MODEL.get(backend_model, {})
        )

        zmin, zmax = bounds["z"]
        redshift_max = zmax

        # Redshifts in range
        zvals = zs
        zvals_inrange = zvals[(zvals >= zmin) & (zvals <= zmax)]

        # common cosmology block
        Omega_nu_real = self.background.mnu / 93.14 / (self.background.H0 / 100.0) ** 2
        Omega_m = self.background.Omega_cdm0 + self.background.Omega_b0 + Omega_nu_real

        # Fixing massive neutrinos to 0 for the boost as decided in TH1-4 telecon, 23.03.2026 but keep same total Om
        Omega_nu = (
            1e-10  # self.background.mnu / 93.14 / (self.background.H0 / 100.0) ** 2
        )

        # defaults for MG/DE parameters (overridden below per model)
        w0 = self.background.w0
        wa = self.background.wa

        # Only enforce LCDM background if not in one of the "allowed" DE models
        if self.gravity_model not in ["wCDM", "w0waCDM", "ide", "mu", "ds"]:
            assert self.background.w0 == -1.0 and self.background.wa == 0.0, (
                "All other emulators are trained for ΛCDM background"
            )

        xi = mgpars.get("xi", 0.0)

        # Define the dictionary to be fed to the emulator
        # Add parameters as necessary
        # EDIT: if new model available, add parameters here
        self.params_mg_emu = {
            "Omega_m": Omega_m,
            "Omega_b": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "H0": self.background.H0,
            "Omega_nu": Omega_nu,
            # MG-family keys
            "fR0": mgpars.get("fr0", 1e-10),
            "omegarc": mgpars.get("omega_rc", 0.0),
            "gamma": mgpars.get("gamma0", 0.55),
            "mu0": mgpars.get("mu0", 0.0),
            "sigma0": mgpars.get("sigma0", 0.0),
            # Scale dependent parameters for mu(k;z)
            "c1": mgpars.get("c1", 0.0),
            "lam": mgpars.get("lam", 0.0),
            # Screening parameters
            "q1": mgpars.get("q1", 0.0),
            "q2": mgpars.get("q2", 0.0),
            "q3": mgpars.get("q3", 0.0),
            # DS-family keys
            "w0": w0,
            "wa": wa,
            "xi": xi,
            # redshift vector
            "z": zvals,
        }

        # -------------------------------
        # Copy parameter dict and clip/tile for in-range z only
        # -------------------------------
        params_inrange = {}
        for key, (lo, hi) in bounds.items():
            if key == "z":
                params_inrange["z"] = zvals_inrange
                continue
            val = self.params_mg_emu.get(key, None)
            if val is None:
                continue
            val_clipped = np.clip(val, lo, hi)
            params_inrange[key] = np.tile(val_clipped, len(zvals_inrange))

        # Compute only for z ≤ redshift_max
        k_emu, boost_inrange = MG_emu.get_nonlinear_boost(**params_inrange)
        # Change h/Mpc --> 1/Mpc
        k_emu *= self.background.h

        # Create the interpolator over the original grid
        boost_inrange_interp = RectBivariateSpline(
            zvals_inrange, k_emu, boost_inrange, kx=1, ky=1
        )

        # ---- choose redshift extrapolation policy ----
        # options: "power_law", "freeze", "one", "taper"
        # validate & store policy
        if high_z_policy not in {"power_law", "freeze", "one", "taper"}:
            raise ValueError("Invalid high_z_policy ...")

        self.z_decay = float(z_decay)

        # Low k extrapolation
        # constant extrapolation of the boost to low  k
        kmin = k_emu[0]
        kmax = k_emu[-1]

        k_low_mask = linearperturbations.k < kmax
        k_target = linearperturbations.k[k_low_mask]

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
            z_out = np.asarray(zvals)
            boost_out = np.empty((len(z_out), len(k_out)))

            idx_in = (z_out >= zmin) & (z_out <= redshift_max)
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
                    -dz / z_decay
                )

            else:
                raise ValueError(f"Unknown high_z_policy: {high_z_policy}")

        # ✅ Apply boost = 1 for z > redshift_max if it is lower than 1 (this is safest thing to do since the extend_spectra results in unphysical high-z extrapolations)
        enforce_floor_highz = True
        if enforce_floor_highz:
            hi = z_out > redshift_max  # z_max = 2 in current setup
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


class BoostedPerturbations:
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

        self.background = (
            base_lin_perturbations.background
        )  # not base_perturbations as those are only LCDM
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
        """Return the boosted nonlinear matter power spectrum.

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
