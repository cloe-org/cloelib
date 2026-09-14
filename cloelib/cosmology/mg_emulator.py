"""Implementation of modified gravity perturbations using the binned MGEmulator.

This module implements nonlinear matter power spectrum perturbations in modified
gravity using an emulator-based boost approach. The emulator predicts the ratio

    P_MG(k, z) / P_LCDM(k, z)

which is then combined with external LCDM nonlinear matter power spectrum predictions
to obtain the full modified gravity matter power spectrum.

The implementation is compatible with the ``cloelib`` perturbation interface
and is designed for use in Euclid-like large-scale structure analyses.

The emulator supports binned modifications of gravity parameterized by the
parameters ``mu`` and ``eta`` within predefined redshift bins.
"""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
# from cloelib.emulator_MG_binned.mg_emulator import MGPerturbations

# General imports
import numpy as np
from scipy import interpolate

# MG emulator import
try:
    from emulator import MGEmulator
except ImportError:
    raise ImportError("MGEmulator could not be imported or initialised.")


class MGPerturbations:
    """Modified gravity perturbations using the MGEmulator boost model.

    The nonlinear matter power spectrum boost predicted by the MGEmulator,

    .. math::

        B(k, z) = \\frac{P_{\\rm MG}(k, z)}{P_{\\Lambda\\rm CDM}(k, z)}

    is combined with external LCDM linear and nonlinear matter power spectrum
    predictions in order to construct the full modified gravity matter power
    spectrum.

    The implementation supports scale-dependent growth quantities and
    redshift-binned modified gravity parameters.

    Notes
    -----
    The emulator is loaded lazily and cached globally such that repeated
    likelihood evaluations reuse the same emulator instance. This significantly
    reduces initialization overhead during MCMC or nested sampling analyses.
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
    ) -> None:
        """Initialize the modified gravity perturbation instance.

        Parameters
        ----------
        background : Background
            Background cosmology object providing cosmological parameters.

        linearperturbations : Perturbations
            Linear LCDM perturbation object used as the baseline linear matter
            power spectrum.

        nonlinearperturbations : Perturbations
            Nonlinear LCDM perturbation object used as the baseline nonlinear
            matter power spectrum.

        redshifts : np.ndarray
            Redshift values used for emulator evaluation and interpolation.

        mu : float
            Modified gravity parameter controlling the effective modification
            to the Poisson equation.

        eta : float
            Modified gravity parameter controlling the gravitational slip.

        bin_index : int
            Index of the active modified gravity redshift bin.

        model_dir : str, optional
            Path to the directory containing the emulator model files.

        Notes
        -----
        The emulator predicts a multiplicative boost relative to LCDM:

        .. math::

            P_{\\rm MG}(k, z)
            =
            B(k, z)
            \\times
            P_{\\Lambda\\rm CDM}(k, z)

        Linear and nonlinear boosts are treated separately using dedicated
        emulator branches.

        All calculations currently assume a spatially flat cosmology.
        """
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
            "Omega_m": (self.background.Omega_cdm0 + self.background.Omega_b0),
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

            MGPerturbations._shared_mg_emu = MGEmulator(model_dir=model_dir)

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

        self.k_linear, boost_linear = self.mg_emu.emulator.linear.predict_boost(
            self.cosmo,
            mu=self.mu,
            eta=self.eta,
            bin_index=self.bin_index,
            zs=self.z,
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
        """Compute the nonlinear matter power spectrum.

        Parameters
        ----------
        zs : array_like
        Redshift values.

        ks : array_like
        Wavenumber values in units of h/Mpc.

        Returns
        -------
        np.ndarray
        Nonlinear matter power spectrum evaluated at the input
        redshift and wavenumber values.
        """
        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Compute the scale-dependent linear growth factor.

        The growth factor is computed from the linear modified gravity
        matter power spectrum according to

        .. math::

        D(z, k)
        =
        \sqrt{
            \frac{
                P_{\rm lin}(z, k)
            }{
                P_{\rm lin}(0, k)
            }
        }

        Parameters
        ----------
            zs : array_like
            Redshift values.

            ks : array_like
        Wavenumber values in units of h/Mpc.

        Returns
        -------
        np.ndarray
        Scale-dependent linear growth factor.
        """

        return np.sqrt(self.Pk_linear_interp(zs, ks) / self.Pk_linear_interp(0.0, ks))

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
        pk0 = self.Pk_linear_interp(np.array([0.0]), k).flatten()

        # ----------------------------------
        # Top-hat window
        # ----------------------------------

        R = 8.0  # Mpc/h
        x = k * R

        # avoid division issues near x=0
        W = np.ones_like(x)

        mask = x > 1e-8

        xm = x[mask]

        W[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / xm**3

        # ----------------------------------
        # sigma8 integral
        # ----------------------------------

        integrand = k**2 * pk0 * W**2

        sigma8_sq = 1.0 / (2.0 * np.pi**2) * np.trapezoid(integrand, k)

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
        r"""Return the modified lensing parameter :math:`\Sigma(z)`.

        The lensing modification parameter is defined as

        .. math::

        \Sigma = \frac{\mu(1 + \eta)}{2}

        within the active modified gravity redshift bin.

        Outside the active bin,

        .. math::

        \Sigma = 1

        corresponding to standard LCDM.

        Parameters
        ----------
        zs : array_like
        Redshift values.

        Returns
        -------
        np.ndarray
        Lensing modification parameter evaluated at the
        input redshift values.
        """

        zs = np.atleast_1d(zs)

        # Default LCDM value everywhere
        sigma = np.ones_like(zs, dtype=float)

        # Active MG bin mask
        mask = self._get_active_bin_mask(zs)

        # MG value only inside active bin
        sigma[mask] = self.mu * (1.0 + self.eta) / 2.0

        return sigma
