"""Implementation of Background and Perturbation cosmology using HMcode2020Emu."""

# cloelib imports
from cloelib.cosmology.cosmology import (
    Background,
    BaryonBoostMixin,
    WithLinearSpectrumGrid,
)
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.auxiliary.math_utils import ensure_z_zero_included

from scipy import interpolate

# General imports
import numpy as np
from typing import Optional, Sequence

# Cosmology imports
try:
    import HMcode2020Emu as hmcodeemu

    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = HM2020_emu.emulator["linear"]["bounds"]["z"][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


class HMemuLinearPerturbations:
    """Class for perturbations cosmology using HMemu, compatibly with the Perturbations protocol."""

    def __init__(self, background: Background, redshifts: np.ndarray):
        """Intialize the HMemuLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        self.z = ensure_z_zero_included(redshifts[redshifts <= redshift_max])
        self.background = background

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
        }

        self._cosmo_params_hm_emu = dict(self.params_hm_emu)

        hm_bounds = HM2020_emu.emulator["linear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 lin emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

        self.params_hm_emu["z"] = self.z

        _, Pk = HM2020_emu.get_linear_pk(**self.params_hm_emu)
        _, Pk_cb = HM2020_emu.get_linear_pk(nonu=True, **self.params_hm_emu)

        k_emu = HM2020_emu.emulator["linear"]["k"] * self.background.h

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_emu,
            self.z,
            self.background.h**-3 * Pk,
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_cb_out = extend_spectra(
            k_emu,
            self.z,
            self.background.h**-3 * Pk_cb,
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out
        self.Pk_cb = Pk_cb_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)

        self.Pk_interp = pk_interp

        pk_cb_interp = interpolate.RectBivariateSpline(
            self.z, self.k, Pk_cb_out, kx=1, ky=1
        )

        self.Pk_cb_interp = pk_cb_interp

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        r"""Compute the linear matter power spectrum.

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts
            hubble_units (Optional[bool]): Flag to specify if output in h units
            k_hunit (Optional[bool]): Flag to specify if wavenumber in h units

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        if k_hunit:
            k_in = ks * self.background.h
        else:
            k_in = ks
        if hubble_units:
            return self.Pk_interp(zs, k_in) * self.background.h**3
        else:
            return self.Pk_interp(zs, k_in)

    def matter_power_spectrum_cb(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        r"""Compute the linear matter power spectrum of cold dark matter + baryons (no neutrinos).

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts
            hubble_units (Optional[bool]): Flag to specify if output in h units
            k_hunit (Optional[bool]): Flag to specify if wavenumber in h units

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        if k_hunit:
            k_in = ks * self.background.h
        else:
            k_in = ks
        if hubble_units:
            return self.Pk_cb_interp(zs, k_in) * self.background.h**3
        else:
            return self.Pk_cb_interp(zs, k_in)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.array([0.0]), ks)
        )

        return D_z_k

    def growth_factor_cb(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for cb for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta_{cb}\delta_{cb}}(z, k)\
            /P_{\rm \delta_{cb}\delta_{cb}}(z=0, k)}\\
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        D_cb_z_k = np.sqrt(
            self.matter_power_spectrum_cb(zs, ks)
            / self.matter_power_spectrum_cb(np.array([0.0]), ks)
        )

        return D_cb_z_k

    def growth_rate(self, zs=None, ks=None) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Args:
            zs (Optional[array_like]): Redshifts at which to calculate the growth rate.
                Defaults to the redshift grid of this instance.
            ks (Optional[array_like]): Wavenumbers at which to calculate the growth rate.
                The HMcode2020Emu growth rate is scale independent, so these only
                set the shape of the output.

        Returns:
            (np.ndarray): The growth rate, with shape (nz,) if ks is None
                and (nz, nk) otherwise.
        """
        return _growth_rate(
            self._cosmo_params_hm_emu,
            self.z if zs is None else zs,
            ks,
        )

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        self.sigma8, _ = HM2020_emu.get_sigma8(**self.params_hm_emu)
        return self.sigma8[0]


class HMemuNonLinearPerturbations:
    """Class for non linear perturbations cosmology using HMemu,  compatibly with the Perturbations protocol."""

    def __init__(
        self,
        background: Background,
        linearperturbations: WithLinearSpectrumGrid,
        redshifts: np.ndarray,
        log10TAGN: Optional[float] = None,
    ):
        """Initialize the HMemuNonLinearPerturbations instance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = HM2020_emu.emulator["nonlinear"]["bounds"]["z"][1]

        self.z = ensure_z_zero_included(redshifts[redshifts <= redshift_max])
        self.background = background
        # Retained so downstream consumers that need the *linear* Pk (e.g.
        # a perturbation-theory backend, which is only valid starting from
        # linear input) can get back to it from a tracer's own (nonlinear)
        # `perturbations` without the caller separately tracking both
        # objects - same attribute name/pattern already used by
        # `EE2NonLinearPerturbations`, `BACCOemuNonLinearPerturbations`,
        # `EmantisFofrNonLinearPerturbations`, and `JAXNonLinearPerturbations`.
        self.linearperturbations = linearperturbations

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
        }
        baryonic_boost = log10TAGN is not None

        if baryonic_boost:
            self.params_hm_emu["log10TAGN"] = log10TAGN

        self._cosmo_params_hm_emu = dict(self.params_hm_emu)

        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                raise ValueError("HMcode 2020 NL emulator out of range.")
            else:
                self.params_hm_emu[key] = np.tile(self.params_hm_emu[key], len(self.z))

        self.params_hm_emu["z"] = self.z

        _, Pk = HM2020_emu.get_nonlinear_pk(
            nonu=False, **self.params_hm_emu, baryonic_boost=baryonic_boost
        )

        _, Pk_cb = HM2020_emu.get_nonlinear_pk(
            nonu=True, **self.params_hm_emu, baryonic_boost=baryonic_boost
        )

        k_emu = HM2020_emu.emulator["nonlinear"]["k"] * self.background.h

        # Low-k extrapolation.
        # Done this way to use Pk array instead of calling an interpolator
        # This only works if the redshift array is exactly the same within
        # range. This should be, but we should probably make sure in some way
        Pk_lin_mask_k = linearperturbations.k < k_emu[0]
        Pk_lin_mask_z = linearperturbations.z <= redshift_max
        Pk_lin = linearperturbations.Pk[Pk_lin_mask_z][:, Pk_lin_mask_k]
        Pk_cb_lin = linearperturbations.Pk_cb[Pk_lin_mask_z][:, Pk_lin_mask_k]
        k_all = np.concatenate((linearperturbations.k[Pk_lin_mask_k], k_emu))
        Pk_all = np.concatenate((Pk_lin, self.background.h**-3 * Pk), axis=1)
        Pk_cb_all = np.concatenate((Pk_cb_lin, self.background.h**-3 * Pk_cb), axis=1)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_all,
            self.z,
            Pk_all,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        k_out, z_out, Pk_cb_out = extend_spectra(
            k_all,
            self.z,
            Pk_cb_all,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        # Alternative method using interpolators
        # Pk_lin = linearperturbations.Pk_interp(self.z, k_emu)
        # k_out, z_out, boost_out = \
        #     extend_spectra(k_emu, self.z, self.background.h ** -3 * Pk / Pk_lin,
        #                    flag_range=True,
        #                    option_wavenumber="power_law",
        #                    option_redshift="power_law", extrap_z = redshifts,
        #                    option_cosmo="const", ns=self.background.ns)
        # self.Pk = linearperturbations.Pk_interp(self.z, k_out) * boost_out

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out
        self.Pk_cb = Pk_cb_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, self.Pk, kx=1, ky=1)

        self.Pk_interp = pk_interp

        pk_cb_interp = interpolate.RectBivariateSpline(
            self.z, self.k, self.Pk_cb, kx=1, ky=1
        )

        self.Pk_cb_interp = pk_cb_interp

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Compute the nonlinear matter power spectrum.

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        return self.Pk_interp(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        r"""Compute the nonlinear matter power spectrum.

        Args:
            ks (numpy.ndarray): Wave number in h Mpc^{-1}
            zs (numpy.ndarray): redshifts

        Returns:
            pk (numpy.ndarray): Linear matter power spectrum at the specified scale and redshift

        """
        return self.Pk_cb_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks)
            / self.matter_power_spectrum(np.array([0.0]), ks)
        )

        return D_z_k

    def growth_factor_cb(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for cb for given redshifts and wavenumbers.

        $$
            D(z, k) =\sqrt{P_{\rm \delta_{cb}\delta_{cb}}(z, k)\
            /P_{\rm \delta_{cb}\delta_{cb}}(z=0, k)}\\
        $$

        and normalizes as for $D(z)/D(0)$.

        Args:
            zs (array_like): Redshifts at which to calculate the growth factor.
            ks (array_like): Wavenumbers at which to calculate the growth factor.

        Returns:
            (np.ndarray): The growth factor as a function of redshift and wavenumber.
        """
        D_cb_z_k = np.sqrt(
            self.matter_power_spectrum_cb(zs, ks)
            / self.matter_power_spectrum_cb(np.array([0.0]), ks)
        )

        return D_cb_z_k

    def growth_rate(self, zs=None, ks=None) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Args:
            zs (Optional[array_like]): Redshifts at which to calculate the growth rate.
                Defaults to the redshift grid of this instance.
            ks (Optional[array_like]): Wavenumbers at which to calculate the growth rate.
                The HMcode2020Emu growth rate is scale independent, so these only
                set the shape of the output.

        Returns:
            (np.ndarray): The growth rate, with shape (nz,) if ks is None
                and (nz, nk) otherwise.
        """
        return _growth_rate(
            self._cosmo_params_hm_emu,
            self.z if zs is None else zs,
            ks,
        )

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """

        self.sigma8, _ = HM2020_emu.get_sigma8(**self.params_hm_emu)
        return self.sigma8[0]


def _growth_rate(params: dict, zs, ks=None) -> np.ndarray:
    r"""Evaluate the HMcode2020Emu growth rate $f = f\sigma_8 / \sigma_8$.

    Redshifts above the range of the emulator are evaluated at its maximum
    redshift, where the growth rate is already close to its matter-domination
    value of unity.

    Parameters
    ----------
    params: dict
        Cosmological parameters of the emulator, one value per parameter.
    zs: array_like
        Redshifts at which to evaluate the growth rate.
    ks: Optional[array_like]
        Wavenumbers used to broadcast the scale-independent growth rate.

    Returns
    -------
    np.ndarray
        The growth rate, with shape (nz,) if ks is None and (nz, nk) otherwise.
    """
    z_max = HM2020_emu.emulator["sigma8"]["bounds"]["z"][1]
    z = np.clip(np.atleast_1d(np.asarray(zs, dtype=float)), 0.0, z_max)
    sigma8, fsigma8 = HM2020_emu.get_sigma8(
        **{key: np.tile(value, len(z)) for key, value in params.items()}, z=z
    )
    f = fsigma8 / sigma8
    if ks is None:
        return f
    return np.tile(f[:, None], (1, np.size(ks)))


def _set_neutrino_masses(background: Background) -> float:
    r"""Set neutrino masses in the parameters dictionary.

    This method adds neutrino masses to the provided dictionary.
    It also ensures consistency with the background cosmology.
    HMcode2020Emu only supports a single species massive of neutrinos, so this method
    throws an error if multiple massive neutrino species are provided.

    Parameters
    ----------
    background: Background
        Background class containing cosmology and background quantities

    Returns
    -------
    float
        The total neutrino mass in eV.
    """
    if background.N_mnu > 1:
        raise ValueError(
            "HMcode2020Emu only supports a single species of neutrinos. "
            "Set N_mnu=1 in the Background class."
        )
    if not np.isclose(background.N_ur, 2.0308, rtol=1e-4):
        raise ValueError(
            "HMcode2020Emu only supports a fixed number of relativistic species (N_ur=2.0308). "
            "Set N_ur=2.0308 in the Background class."
            "[Note that HMcode2020Emu actually sets N_ur=2.0328,"
            "this will be fixed in a future release.]"
        )
    if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
        raise ValueError(
            "HMcode2020Emu only supports a fixed number of effective"
            f"relativistic species (N_eff=3.044). Found {background.N_eff} "
            "Ensure that N_eff=3.044 in the Background class."
            "[Note that HMcode2020Emu actually sets N_eff=3.046,"
            "this will be fixed in a future release.]"
        )
    if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
        raise ValueError(
            "HMcode2020Emu only supports a single species of neutrinos. "
            "Set N_mnu=1 in the Background class."
        )
    else:
        mnu_arg = float(background.mnu)
    # returns the neutrino mass in eV
    return mnu_arg


class HMcode2020BaryonBoostMixin(BaryonBoostMixin):
    """Mixin providing baryonic suppression via an HMcode2020-computed P_baryon/P_dmo ratio.

    Can be combined with *any* nonlinear perturbations class (not just HMcode2020emu)::

        class MyPert(HMcode2020BaryonBoostMixin, HMemuNonLinearPerturbations):
            def __init__(self, background, linearperturbations, redshifts, log10TAGN=7.8):
                HMemuNonLinearPerturbations.__init__(
                    self, background, linearperturbations, redshifts
                )
                HMcode2020BaryonBoostMixin.__init__(self, log10TAGN=log10TAGN)

    Or use :func:`~cloelib.cosmology.cosmology.with_baryon_boost`.

    .. note::
        ``__init__`` must be called **after** the base NonLinear ``__init__``
        because it reads ``self.background`` and ``self.z``.
        Scales below the nonlinear emulator k range get suppression = 1.
    """

    #: Both supplied by the nonlinear perturbations class this mixin is composed with.
    background: Background
    z: np.ndarray

    def __init__(self, log10TAGN: float) -> None:
        """Initialise the HMcode2020 baryon-ratio spline.

        Runs the emulator twice (DMO and baryonic) and stores
        ``B(z, k)`` as a bivariate spline in ``self._baryon_ratio_interp``.

        Call this **after** the base NonLinear ``__init__`` so that
        ``self.background`` and ``self.z`` are already set. The emulator
        parameters are built from ``self.background`` rather than taken from
        the base class, so the HMcode2020 baryonic boost can be applied on top
        of a different nonlinear prescription.

        Parameters
        ----------
        log10TAGN:
            log₁₀ of the AGN heating temperature in Kelvin.
            Typical range: 7.6 (weak) – 8.3 (strong feedback).
        """
        # The emulator parameters are built here rather than taken from the
        # base class, so this mixin also works on top of a non-HMcode2020emu
        # nonlinear prescription.
        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]
        cosmo_params = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            "As": self.background.As,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": _set_neutrino_masses(self.background),
            "w0": self.background.w0,
            "wa": self.background.wa,
            "log10TAGN": log10TAGN,
        }
        for key, value in cosmo_params.items():
            if np.prod(value - hm_bounds[key]) > 0:
                raise ValueError(
                    f"HMcode 2020 NL emulator: {key}={value} is out of bounds "
                    f"{hm_bounds[key]}."
                )
        z_emu = np.unique(self.z[self.z <= hm_bounds["z"][1]])
        params_baryon = {
            key: np.tile(value, len(z_emu)) for key, value in cosmo_params.items()
        }
        params_baryon["z"] = z_emu
        params_dmo = {k: v for k, v in params_baryon.items() if k != "log10TAGN"}
        _, Pk_dmo = HM2020_emu.get_nonlinear_pk(
            nonu=False, **params_dmo, baryonic_boost=False
        )
        _, Pk_baryon = HM2020_emu.get_nonlinear_pk(
            nonu=False, **params_baryon, baryonic_boost=True
        )
        ratio = Pk_baryon / Pk_dmo
        k_nl_phys = (
            HM2020_emu.emulator["nonlinear"]["k"] * self.background.h
        )  # h/Mpc -> 1/Mpc
        self._baryon_k_nl_min = k_nl_phys[0]
        self._baryon_k_range = (float(k_nl_phys[0]), float(k_nl_phys[-1]))

        # Extend with a power law before interpolating, and interpolate in
        # log k, as `EE2NonLinearPerturbations` does for the nonlinear boost.
        # Without this the spline holds its boundary value outside the emulator
        # range, giving an unphysical flat tail.
        k_out, z_out, ratio_out = extend_spectra(
            k_nl_phys,
            z_emu,
            ratio,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=z_emu,
            option_cosmo="const",
            ns=self.background.ns,
        )
        # Interpolated log-log: log B against log k.
        self._baryon_ratio_interp = interpolate.RectBivariateSpline(
            z_out, np.log(k_out), np.log(ratio_out), kx=1, ky=1
        )

    @property
    def baryon_k_range(self) -> tuple[float, float]:
        """Wavenumbers in 1/Mpc over which ``baryonic_suppression`` is emulated.

        Below the range the suppression is set to 1; above it the spline holds
        its boundary value rather than extrapolating.
        """
        return self._baryon_k_range

    def baryonic_suppression(self, zs, ks, k_hunit: bool = False) -> np.ndarray:
        """Return B(z, k) = P_baryon(z, k) / P_dmo(z, k).

        Parameters
        ----------
        zs : array_like
            Redshifts.
        ks : array_like
            Wavenumbers. By default in 1/Mpc; pass k_hunit=True for h/Mpc.
        k_hunit : bool
            If True, convert ks from h/Mpc to 1/Mpc before evaluating.

        Returns
        -------
        np.ndarray
            Baryonic suppression factor, shape (nz, nk).
            Returns 1.0 for scales below the nonlinear emulator k range.
        """
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        if k_hunit:
            ks = ks * self.background.h
        result = np.ones((len(zs), len(ks)))
        in_range = ks >= self._baryon_k_nl_min
        if np.any(in_range):
            result[:, in_range] = np.exp(
                self._baryon_ratio_interp(zs, np.log(ks[in_range]))
            )
        return result
