"""Implementation of Background and Perturbation cosmology using CLASS."""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.extrapolator import extend_spectra

# General imports
import numpy as np
import copy
from typing import Optional, Union, Sequence
from scipy import interpolate
import DMemu

# Cosmology imports
try:
    from classy import Class  # type: ignore
except ImportError as e:
    raise ImportError("classy could not be imported.") from e


import os
import sys
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# TODO : implement classes for Background and LinearPerturbations
# (needs interface with classy corresponding to my modified version of CLASS)

class tbDDMNonLinearPerturbations:
    """Class for non-linear perturbations cosmology using CLASS, inheriting from Perturbations parent class."""

    c0 = SPEED_OF_LIGHT / 1000
    _emul_cache = None  # shared DMemu instance — loaded once, reused across all instantiations
    
    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        f_dcdm: float,
        epsilon: float,
        Gamma: float,
        use_emulator: bool = False,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the tbDDMNonLinearPerturbations instance."""
        self.background   = background
        self.z            = redshifts
        self.kmax         = 40
        self.f            = f_dcdm
        self.Gamma        = Gamma
        self.vk           = epsilon * tbDDMNonLinearPerturbations.c0
        self.use_emulator = use_emulator
        self.log10TAGN    = log10TAGN if log10TAGN is not None else 7.6

        # Initialize NN emulator (load once, reuse via class-level cache)
        if tbDDMNonLinearPerturbations._emul_cache is None:
            tbDDMNonLinearPerturbations._emul_cache = DMemu.TBDemu()
        self.emul = tbDDMNonLinearPerturbations._emul_cache

        # ---- NL LCDM baseline Pk ----------------------------------------
        if not use_emulator:
            # CLASS path: halofit-LCDM
            # (boost is defined relative to halofit-LCDM by Bucko et al.)
            self.interface_args = copy.deepcopy(self.background.interface_args)
            self.interface_args["CLASSparams"]["output"] = "mPk, mTk"
            self.interface_args["CLASSparams"]["P_k_max_1/Mpc"] = self.kmax
            self.interface_args["CLASSparams"]["z_max_pk"] = np.max(self.z)
            self.interface_args["CLASSparams"]["non linear"] = "halofit"
            self.results = Class()
            self.results.set(self.interface_args["CLASSparams"])
            self.results.compute()

            _, self.k, _ = self.results.get_pk_and_k_and_z(
                nonlinear=True, only_clustering_species=False, h_units=False
            )
            pk_nl_on_z = np.array(
                [[self.results.pk(ki, zi) for ki in self.k] for zi in self.z]
            )  # (nz, nk)

        else:
            # Emulator path: cosmopower-JAX HMcode2020 (w0wa-3degen-nonlinear.npz).
            # Note: the 2bDDM boost was trained vs halofit-LCDM, but HMcode2020 and
            # halofit agree at the few-percent level, so the inconsistency is small.
            from cloelib.cosmology.cosmopower_jax_cosmology import (
                emulator_data, load_pk_emulator, k_modes_path
            )
            cp_NL  = load_pk_emulator(emulator_data("w0wa-3degen-nonlinear.npz"))
            k_emu  = np.loadtxt(k_modes_path)
            self.k = k_emu

            h         = self.background.h
            # For CLASSBackground, background.mnu is always the total neutrino mass
            mnu_total = self.background.mnu

            params_nl = {
                "ombh2":    np.tile(self.background.Omega_b0 * h**2,        len(self.z)),
                "omch2":    np.tile(self.background.Omega_cdm0 * h**2,      len(self.z)),
                "H0":       np.tile(self.background.H0,                     len(self.z)),
                "ns":       np.tile(self.background.ns,                     len(self.z)),
                "lnAs":     np.tile(np.log(self.background.As * 1e10),      len(self.z)),
                "w0":       np.tile(self.background.w0,                     len(self.z)),
                "wa":       np.tile(self.background.wa,                     len(self.z)),
                "mnu":      np.tile(mnu_total,                              len(self.z)),
                "logT_AGN": np.tile(self.log10TAGN,                         len(self.z)),
                "z":        self.z,
            }
            pk_nl_on_z = np.array(cp_NL.predict(params_nl))  # (nz, nk)

        # ---- 2bDDM boost (same for both paths) ---------------------------
        # Single batched emulator call over the full (self.z × self.k) grid.
        kk, zz    = np.meshgrid(self.k, self.z)
        k_flat    = kk.ravel()
        z_flat    = zz.ravel()

        boost_flat = np.ones(len(z_flat))          # default 1 for z > 2.35
        mask_valid = z_flat <= 2.35
        if np.any(mask_valid):
            with SuppressOutput():
                boost_vals = self.emul.predict(
                    k_flat[mask_valid],
                    z_flat[mask_valid],
                    self.f, self.vk, self.Gamma,
                    allow_z_extrapolation=False
                )
            boost_flat[mask_valid] = np.asarray(boost_vals).ravel()

        boost_grid = boost_flat.reshape(len(self.z), len(self.k))
        pk_2bddm   = pk_nl_on_z * boost_grid

        # Extend Pk to k=500 1/Mpc to avoid Akima extrapolation blowup at low z
        k_out, z_out, Pk_out = extend_spectra(
            self.k, self.z, pk_2bddm,
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=self.z,
            option_cosmo="const",
            ns=self.background.ns,
            extrap_kmax=500.0,
        )
        self.k = k_out

        # Pre-computed spline: matter_power_spectrum is a fast lookup
        self.Pk_int = interpolate.RectBivariateSpline(
            z_out, k_out, Pk_out, kx=1, ky=1
        )

    def boost_2bDDM(self, z, k) -> float:
        """Calculate the boost factor for the 2bDDM suppression, with the emulator by Bucko et al. (2307.03222)
        
        Parameters
        ----------
        z: float
           redshift
        k: float
           wavenumber

        Returns
        -------
        S_2bDDM: float
                boost factor for the 2bDDM suppression at given redshift and wavenumber
        """
        if (z > 2.35):
            return 1.0 # we shouldn't extrapolate the emulator beyond the training domain for z
                          # Extrapolation for kappa > 6 h/Mpc is done by adding a constant suppression continuously attached
                          # to the one provided by an emulator
        else:
            with SuppressOutput():
                S_2bDDM = self.emul.predict(np.asarray(k).reshape(-1),
                                            np.asarray(z).reshape(-1),
                                            self.f,
                                            self.vk,
                                            self.Gamma, 
                                            allow_z_extrapolation = False)
        
            return np.asarray(S_2bDDM).item()


    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the CLASS non-linear matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk: numpy.ndarray
            Non-linear matter power spectrum at the specified scale
            and redshift
        """
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")

        return self.Pk_int(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        Returns:
        --------
        np.ndarray
            The growth factor at the specified redshift and wavenumber.
        """
        # NOTE: a priori here we should use the "linear" matter power spectrum for 2bDDM
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.zeros_like(zs), ks)
        )

        return D_z_k

    def growth_rate(self) -> np.ndarray:
        """
        Calculate the growth rate f(z).

        Returns
        -------
        np.ndarray
            Scale-independent growth rate f(z)
        """
        arr = [self.background.scale_independent_growth_factor_f(zi) for zi in self.z]  # type: ignore[union-attr]
        return np.array(arr)
