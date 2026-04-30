"""Implementation of modified gravity boost perturbations using MGEmulator."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
#from cloelib.emulator_MG_binned.mg_emulator import MGPerturbations

# General imports
import numpy as np
from scipy import interpolate

# MG emulator import
try:
    from emulator import MGEmulator
except ImportError:
    raise ImportError("MGEmulator could not be imported or initialised.")


class MGPerturbations:
    """
    Modified gravity perturbations wrapper compatible with the Perturbations protocol.
    """

    # -------------------------------------------------
    # Shared emulator cache
    # Loaded only once for the full Python session
    # -------------------------------------------------

    _shared_mg_emu = None

    def __init__(
        self,
        background: Background,
        linearperturbations: Perturbations,
        nonlinearperturbations: Perturbations,
        redshifts: np.ndarray,
        mu: float,
        eta: float,
        bin_index: int,
        model_dir: str = "./models",
    ):

        assert background.Omega_k0 == 0, "Non-flat geometries not supported"

        self.background = background
        self.linearperturbations = linearperturbations
        self.nonlinearperturbations = nonlinearperturbations
        self.z = np.asarray(redshifts)
        self.mu = mu
        self.eta = eta
        self.bin_index = int(bin_index)

        # ----------------------------------
        # Convert cloelib background
        # -> emulator cosmology dictionary
        # ----------------------------------

        self.cosmo = {
            "Omega_m": (
                self.background.Omega_cdm0
                + self.background.Omega_b0
            ),
            "Omega_b": self.background.Omega_b0,
            "h": self.background.H0 / 100.0,
            "n_s": self.background.ns,
            "A_s": self.background.As,
        }

        # ----------------------------------
        # TRUE lazy loading
        #
        # Only load emulator once.
        # Reuse for all likelihood evaluations.
        # This avoids repeated PCA/GP loading
        # during MCMC.
        # ----------------------------------

        if MGPerturbations._shared_mg_emu is None:
            print("Loading MGEmulator into shared memory...")

            MGPerturbations._shared_mg_emu = MGEmulator(
                model_dir=model_dir
            )

            print("MGEmulator loaded once and cached.")

        self.mg_emu = MGPerturbations._shared_mg_emu

        # ----------------------------------
        # Emulator boost prediction
        # ----------------------------------

        self.k_boost, boost = self.mg_emu.predict_boost(
            self.cosmo,
            mu=self.mu,
            eta=self.eta,
            bin_index=self.bin_index,
            zs=self.z,
        )

        # ----------------------------------
        # Linear boost only (NN branch)
        # ----------------------------------

        self.k_linear, boost_linear = (
            self.mg_emu.emulator.linear.predict_boost(
                self.cosmo,
                mu=self.mu,
                eta=self.eta,
                bin_index=self.bin_index,
                zs=self.z,
            )
        )

        # ----------------------------------
        # Linear LCDM baseline
        # ----------------------------------

        pk_lcdm_linear = self.linearperturbations.Pk_interp(
            self.z,
            self.k_linear,
        )

        # ----------------------------------
        # Final MG linear spectrum
        # ----------------------------------

        self.Pk_linear = boost_linear * pk_lcdm_linear

        self.Pk_linear_interp = interpolate.RectBivariateSpline(
            self.z,
            self.k_linear,
            self.Pk_linear,
            kx=1,
            ky=1,
        )

        # ----------------------------------
        # Nonlinear LCDM baseline
        # ----------------------------------

        pk_lcdm_nl = self.nonlinearperturbations.Pk_interp(
            self.z,
            self.k_boost,
        )

        # ----------------------------------
        # Final MG nonlinear spectrum
        # ----------------------------------

        self.Pk = boost * pk_lcdm_nl
        self.k = self.k_boost

        # ----------------------------------
        # Final interpolator
        # ----------------------------------

        self.Pk_interp = interpolate.RectBivariateSpline(
            self.z,
            self.k,
            self.Pk,
            kx=1,
            ky=1,
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Return matter power spectrum P(k, z)."""
        return self.Pk_interp(zs, ks)

    # ----------------------------------
    # CHANGE 3:
    # growth_factor should use LINEAR P(k)
    # not nonlinear P(k)
    # ----------------------------------

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Return scale-dependent linear growth factor.

        D(z, k) = sqrt(P_lin(z, k) / P_lin(z=0, k))
        """

        return np.sqrt(
            self.Pk_linear_interp(zs, ks) /
            self.Pk_linear_interp(0.0, ks)
        )
    
    # ----------------------------------
    # CHANGE 4:
    # add growth_rate()
    # ----------------------------------

    def growth_rate(self, zs=None, ks=None) -> np.ndarray:
        r"""Return scale-dependent linear growth rate.

        f(z, k) = d ln D / d ln a
            = -(1 + z) d ln D / dz
        """

        if zs is None:
            zs = self.z

        if ks is None:
            ks = self.k_linear

        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)

        D = self.growth_factor(zs, ks)

        lnD = np.log(D)

        dlnD_dz = np.gradient(lnD, zs, axis=0)

        f = -(1.0 + zs[:, None]) * dlnD_dz

        return f
        
    
    # ----------------------------------
    # CHANGE 5:
    # add sigma8_0()
    # ----------------------------------

    def sigma8_0(self) -> float:
        r"""Return modified gravity sigma8 at z = 0.

        Computed directly from the linear MG matter power spectrum:

        sigma8^2 = (1 / 2pi^2) ∫ dk k^2 P_lin(k, z=0) W^2(kR)

        with

        R = 8 Mpc/h

        and spherical top-hat window

        W(x) = 3 [sin(x) - x cos(x)] / x^3
        """

        # ----------------------------------
        # Use linear emulator k-grid
        # ----------------------------------

        k = self.k_linear  # (Nk,)

        # linear MG P(k, z=0)
        pk0 = self.Pk_linear_interp(
            np.array([0.0]),
            k
        ).flatten()

        # ----------------------------------
        # Top-hat window
        # ----------------------------------

        R = 8.0  # Mpc/h
        x = k * R

        # avoid division issues near x=0
        W = np.ones_like(x)

        mask = x > 1e-8

        xm = x[mask]

        W[mask] = (
            3.0
            * (np.sin(xm) - xm * np.cos(xm))
            / xm**3
        )

        # ----------------------------------
        # sigma8 integral
        # ----------------------------------

        integrand = k**2 * pk0 * W**2

        sigma8_sq = (
            1.0
            / (2.0 * np.pi**2)
            * np.trapezoid(integrand, k)
        )

        return np.sqrt(sigma8_sq)
    
        
    def _get_active_bin_mask(self, zs):
        """Return boolean mask for the active MG redshift bin.

        Bin definitions:

            bin_index = 0 : 0.00 < z < 0.43
            bin_index = 1 : 0.43 < z < 0.91
            bin_index = 2 : 0.91 < z < 1.47
            bin_index = 3 : 1.47 < z < 2.15
            bin_index = 4 : 2.15 < z < 3.00
        """

        zs = np.atleast_1d(zs)

        bin_edges = {
            0: (0.00, 0.43),
            1: (0.43, 0.91),
            2: (0.91, 1.47),
            3: (1.47, 2.15),
            4: (2.15, 3.00),
        }

        if self.bin_index not in bin_edges:
            raise ValueError(
                f"Invalid bin_index={self.bin_index}. Must be between 0 and 4."
            )

        zmin, zmax = bin_edges[self.bin_index]

        return (zs > zmin) & (zs <= zmax)
        
    def matter_power_spectrum_cb(self, zs, ks):
        raise NotImplementedError(
            "matter_power_spectrum_cb is not yet implemented for MGPerturbations"
        )


    def Sigma(self, zs):
        r"""Return the modified lensing parameter Sigma(z).

        Defined as:

        :contentReference[oaicite:0]{index=0}

        but only inside the active MG redshift bin.

        Outside the active bin:

        :contentReference[oaicite:1]{index=1}

        corresponding to standard LCDM.
        """

        zs = np.atleast_1d(zs)

        # Default LCDM value everywhere
        sigma = np.ones_like(zs, dtype=float)

        # Active MG bin mask
        mask = self._get_active_bin_mask(zs)

        # MG value only inside active bin
        sigma[mask] = self.mu * (1.0 + self.eta) / 2.0

        return sigma
