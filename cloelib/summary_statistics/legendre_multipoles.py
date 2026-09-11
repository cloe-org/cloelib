"""Module to compute Legendre multipoles."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.observables.spectro import SpectroPower
from cloelib.summary_statistics.APDistortion import APDistortion
from cloelib.auxiliary.math_utils import legendre
from cloelib.auxiliary.fftlog import fftlog

# General imports
import functools
from typing import Optional
import numpy as np
from copy import deepcopy

# cosmolib imports
from cosmolib.data import (
    PowerSpectrumMultipoles,
    PowerSpectrumMultipolesMixingMatrix,
    TwoPointCorrelationMultipoles,
    TwoPointCorrelationPolar,
)


def format_output(stat: str):
    """Decorator to format the output of the main GC observables.

    Parameters
    ----------
    stat: str
        Type of output to format ('PK' or '2PCF').
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            """cosmolib format is returned in Mpc/h units, differently from
            cloelib standards which is in Mpc units
            """

            def get_arg(name, idx):
                return kwargs[name] if name in kwargs else args[idx]

            def set_arg(name, idx, value):
                nonlocal args, kwargs
                if name in kwargs:
                    kwargs[name] = value
                else:
                    args = (*args[:idx], value, *args[idx + 1 :])

            if kwargs.get("format_type") != "cosmolib":
                return func(self, *args, **kwargs)

            h_fid = self.spectro_power.background.h

            if stat == "PK_multipoles":
                if "convolved" in func.__name__:
                    mixing_matrix = get_arg("mixing_matrix", 0)
                    rescaled_mixing_matrix = deepcopy(mixing_matrix)
                    scale_h = mixing_matrix.kout
                    k_center = kwargs.get("k_center", scale_h)
                    for key in [0, 2, 4]:
                        rescaled_mixing_matrix.kin[key] = mixing_matrix.kin[key] * h_fid
                        set_arg("mixing_matrix", 0, rescaled_mixing_matrix)
                else:
                    scale_h = get_arg("k", 0)
                    k_center = scale_h
                    set_arg("k", 0, scale_h * h_fid)
            else:
                scale_h = get_arg("s", 0)
                set_arg("s", 0, scale_h / h_fid)
                if stat == "2PCF_polar":
                    mu = get_arg("mu", 1)

            result = func(self, *args, **kwargs)

            cosmo = {
                key: float(val) if isinstance(val, float) else val
                for key, val in vars(self.spectro_power.background).items()
                if isinstance(val, (float, int)) and key != "h"
            }

            if stat == "PK_multipoles":
                out = (
                    np.array(
                        [
                            result.get(f"ell{i}", np.zeros(len(scale_h)))
                            for i in range(5)
                        ]
                    )
                    * h_fid**3
                )
                return PowerSpectrumMultipoles(
                    k=k_center,
                    keff=scale_h,
                    Nmodes=np.zeros_like(scale_h),
                    multipoles=out,
                    fiducial_cosmology=cosmo,
                    zeff=self.spectro_power.redshift,
                    nbar=self.nbar,
                    Psn=1.0 / self.nbar,
                )
            elif stat == "2PCF_multipoles":
                out = np.array(
                    [result.get(f"ell{i}", np.zeros(len(scale_h))) for i in range(5)]
                )
                return TwoPointCorrelationMultipoles(
                    s=scale_h,
                    multipoles=out,
                    fiducial_cosmology=cosmo,
                    zeff=self.spectro_power.redshift,
                )
            elif stat == "2PCF_polar":
                return TwoPointCorrelationPolar(
                    s=scale_h,
                    mu=mu,
                    correlation=result,
                    fiducial_cosmology=cosmo,
                    zeff=self.spectro_power.redshift,
                )

        return wrapper

    return decorator


class LegendreMultipoles:
    """Class to compute spectroscopic Legendre multipoles of the galaxy power spectrum."""

    def __init__(
        self,
        spectro_power: SpectroPower,
        background_fiducial: Background,
        parameters: dict,
        nbar: float,
    ):
        """Initialize the class instance.

        Args:
            spectro_power (SpectroPower): Class returning the anisotropic power spectrum (only density and
                velocity field couplings; noise and systematics are included directly here)
            background_fiducial (Background): Background class for computing fiducial background distances
            parameters (dict): Dictionary containing shot noise and parameters related to
                observational systematics
            nbar (float): Mean number denisty of the sample
        """
        self.spectro_power = spectro_power
        self.redshift = spectro_power.redshift
        self.background_fiducial = background_fiducial
        self.ap_distortion = APDistortion(spectro_power.background, background_fiducial)

        self.mu_grid, self.mu_weights = np.polynomial.legendre.leggauss(10)
        self.mu_grid = 0.5 * (self.mu_grid + 1.0)
        self.mu_weights *= 0.5

        self.parameters = parameters
        self.nbar = nbar

    def _ensure_array(self, param):
        """Ensure that the input parameter is a NumPy array.

        If the input is a scalar, it is converted to a NumPy array.

        Args:
            param (scalar|array-like): Input parameter.

        Returns:
            (numpy.ndarray): Input parameter as a NumPy array.
        """
        if np.isscalar(param):
            param = np.array([param])
        return np.asarray(param)

    def _k_AP(
        self, k: np.ndarray, mu: np.ndarray, zs: float, use_AP: Optional[bool] = True
    ) -> np.ndarray:
        r"""AP-distorted wavenumber.

        $$
            k(k_{\rm fid},\mu_{\rm fid}, z) &= k_{\rm fid} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{1/2}
        $$
        Parameters:
            k (np.ndarray): Fiducial wavenumber
            mu (np.ndarray): Fiducial angle (cosinus) to the line of sight
            z (float): Redshift
            use_AP (bool): Flag to switch between with and without AP corrections

        Returns:
            kAP (np.ndarray): AP-distorted wavenumber
        """
        q_tr = self.ap_distortion.q_AP_tr(zs) if use_AP else 1.0
        q_lo = self.ap_distortion.q_AP_lo(zs) if use_AP else 1.0
        return np.outer(k, np.sqrt(mu**2 / q_lo**2 + (1.0 - mu**2) / q_tr**2))

    def _mu_AP(
        self, mu: np.ndarray, zs: float, use_AP: Optional[bool] = True
    ) -> np.ndarray:
        r"""AP-distorted angle (cosinus) to the line of sight.

        .. math::
            \mu(\mu_{\rm fid}, z) &= \frac{\mu_{\rm fid}}{q_\parallel(z)} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{-1/2}
        Parameters
        ----------
        mu: np.ndarray
           Fiducial angle (cosinus) to the line of sight
        z: float
           Redshift
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        muAP: np.ndarray
           AP-distorted angle (cosinus) to the line of sight
        """
        q_tr = self.ap_distortion.q_AP_tr(zs) if use_AP else 1.0
        q_lo = self.ap_distortion.q_AP_lo(zs) if use_AP else 1.0
        return mu / q_lo / np.sqrt(mu**2 / q_lo**2 + (1.0 - mu**2) / q_tr**2)

    def _damping_function(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Damping function due to GCsp redshift uncertainty.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        damping_function: np.ndarray
            Damping function due to GCsp redshift uncertainty
        """
        sigma_z = self.parameters["sigmaz"]
        sigma_r = (
            299792.458
            * sigma_z
            / self.background_fiducial.hubble_parameter(self.redshift)
        )
        return np.exp(-(k**2) * mu**2 * sigma_r**2)

    def _Pk2d_noise(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from expansion of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        Pk2d_noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = (
            self.parameters["NP0"] * self._Pk2d_noise_k0(k)
            + self.parameters["NP20"] * self._Pk2d_noise_k2(k)
            + self.parameters["NP22"] * self._Pk2d_noise_k2mu2(k, mu)
        )
        return noise

    def _Pk2d_noise_k0(self, k: np.ndarray) -> np.ndarray:
        r"""Leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = np.full_like(k, 1.0)
        return noise / self.nbar

    def _Pk2d_noise_k2(self, k: np.ndarray) -> np.ndarray:
        r"""Isotropic next-to-leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2
        return noise / self.nbar

    def _Pk2d_noise_k2mu2(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Anisotropic next-to-leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2 * legendre(2, mu)
        return noise / self.nbar

    def _Pk2d_tot(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Total 2D power spectrum (including RSD, systematics, and noise).

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        Pk2d_tot: np.ndarray
            Total 2D power spectrum (including RSD, systematics, and noise)
        """
        return self.spectro_power.Pk2d_rsd(k, mu) * self._damping_function(k, mu) * (
            1.0 - self.parameters["fout"]
        ) ** 2 + self._Pk2d_noise(k, mu)

    def _Pk2d_term_tot(
        self, k: np.ndarray, mu: np.ndarray, term_list: list
    ) -> np.ndarray:
        r"""2D power spectrum of specific RSD terms, accounting for systematics.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        term_list: list
            List of terms to compute
        Returns
        -------
        Pk2d_term_tot: np.ndarray
            2D power spectrum of specific RSD terms
        """
        return (
            self.spectro_power.Pk2d_term_rsd(k, mu, term_list=term_list)
            * self._damping_function(k, mu)
            * (1.0 - self.parameters["fout"]) ** 2
        )

    @format_output("PK_multipoles")
    def power_multipoles(
        self,
        k: np.ndarray,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        format_type: Optional[str] = None,
    ) -> dict:
        r"""Power spectrum Legendre multipoles.

        Parameters:
            k (np.ndarray): Wavenumber
            ells (np.ndarray): Legendre multipole order
            use_AP (bool): Flag to switch between with and without AP corrections
            format_type (str): Type of output format
        Returns:
            multipoles (dict): Power spectrum Legendre multipoles
        """
        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])
        AP_factor = (
            self.ap_distortion.q_AP_tr(self.redshift) ** 2
            * self.ap_distortion.q_AP_lo(self.redshift)
            if use_AP
            else 1.0
        )
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / AP_factor
        kAP = self._k_AP(k, self.mu_grid, self.redshift, use_AP=use_AP)
        muAP = self._mu_AP(self.mu_grid, self.redshift, use_AP=use_AP)
        Pk2d_tot = self._Pk2d_tot(kAP, muAP)
        multipoles = {}
        for i, ell in enumerate(ells):
            leg = legendre(ell, self.mu_grid)
            multipoles[f"ell{ell}"] = np.einsum(
                "ab,b,b->a", Pk2d_tot, leg, self.mu_weights
            )
            multipoles[f"ell{ell}"] *= 2.0 * prefactors[i]

        return multipoles

    def power_term_multipoles(
        self,
        k: np.ndarray,
        term_list: list,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
    ) -> dict:
        r"""Power spectrum Legendre multipoles of specified terms.

        Parameters:
            k (np.ndarray): Wavenumber
            term_list (list): List of terms to compute
            ells (np.ndarray): Legendre multipole order
            use_AP (bool): Flag to switch between with and without AP corrections

        Returns:
            multipoles (dict): Power spectrum Legendre multipoles of specified terms
        """
        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])
        AP_factor = (
            self.ap_distortion.q_AP_tr(self.redshift) ** 2
            * self.ap_distortion.q_AP_lo(self.redshift)
            if use_AP
            else 1.0
        )
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / AP_factor
        kAP = self._k_AP(k, self.mu_grid, self.redshift, use_AP=use_AP)
        muAP = self._mu_AP(self.mu_grid, self.redshift, use_AP=use_AP)
        Pk2d = np.empty((len(term_list), len(k), len(self.mu_grid)))
        rsd_ids = [index for index, term in enumerate(term_list) if "noise" not in term]
        if rsd_ids:
            Pk2d[rsd_ids] = self._Pk2d_term_tot(
                kAP, muAP, term_list=[term_list[index] for index in rsd_ids]
            )
        noise_ids = [index for index in range(len(term_list)) if index not in rsd_ids]
        if noise_ids:
            noise_func = {
                "noise_k0": self._Pk2d_noise_k0,
                "noise_k2": self._Pk2d_noise_k2,
                "noise_k2mu2": self._Pk2d_noise_k2mu2,
            }
            Pk2d[noise_ids] = np.array(
                [
                    (
                        noise_func[term_list[index]](kAP, muAP)
                        if term_list[index] == "noise_k2mu2"
                        else noise_func[term_list[index]](kAP)
                    )
                    for index in noise_ids
                ]
            )
        multipoles = {}
        for i, ell in enumerate(ells):
            leg = legendre(ell, self.mu_grid)
            multipoles[f"ell{ell}"] = np.einsum(
                "abc,c,c->ab", Pk2d, leg, self.mu_weights
            )
            multipoles[f"ell{ell}"] *= 2.0 * prefactors[i]
        return multipoles

    @format_output("PK_multipoles")
    def convolved_power_multipoles(
        self,
        mixing_matrix: PowerSpectrumMultipolesMixingMatrix,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        format_type: Optional[str] = None,
        k_center: Optional[np.ndarray] = None,
    ) -> dict:
        r"""Power spectrum Legendre multipoles convolved with the mixing matrix.

        Parameters:
            mixing_matrix (PowerSpectrumMultipolesMixingMatrix): Dicitonary containing the mixing matrix
            ells (np.ndarray): Legendre multipole order
            use_AP (bool): Flag to switch between with and without AP corrections
            format_type (str): Type of output format

        Returns:
            multipoles_out (dict): Convolved power spectrum Legendre multipoles
        """
        ells_tot = [0, 2, 4]
        ells = self._ensure_array(ells) if ells is not None else ells_tot

        kin_arrays = [mixing_matrix.kin[ell] for ell in ells_tot]

        if all(np.array_equal(kin_arrays[0], kin) for kin in kin_arrays):
            multipoles_in = self.power_multipoles(
                k=kin_arrays[0], ells=ells_tot, use_AP=use_AP
            )
        else:
            multipoles_in = {
                f"ell{ell}": self.power_multipoles(
                    k=kin_arrays[i], ells=[ell], use_AP=use_AP
                )[f"ell{ell}"]
                for i, ell in enumerate(ells_tot)
            }

        multipoles_out = {}
        multipoles_out["k"] = mixing_matrix.kout
        for ell in ells:
            multipoles_out[f"ell{ell}"] = sum(
                np.dot(
                    mixing_matrix.mixing[f"ELL_{ell}-{ell_prime}"],
                    multipoles_in[f"ell{ell_prime}"],
                )
                for ell_prime in ells_tot
            )

        return multipoles_out

    def convolved_power_term_multipoles(
        self,
        mixing_matrix: PowerSpectrumMultipolesMixingMatrix,
        term_list: list,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
    ) -> dict:
        r"""Convolved power spectrum multipoles of specified terms.

        Parameters:
            mixing_matrix (PowerSpectrumMultipolesMixingMatrix): Dicitonary containing the mixing matrix
            term_list (list): List of terms to compute
            ells (np.ndarray): Legendre multipole order
            use_AP (bool): Flag to switch between with and without AP corrections

        Returns:
            multipoles_out (dict): Convolved power spectrum Legendre multipoles of specified terms
        """
        ells_tot = [0, 2, 4]
        ells = self._ensure_array(ells) if ells is not None else ells_tot

        kin_arrays = [mixing_matrix.kin[ell] for ell in ells_tot]

        if all(np.array_equal(kin_arrays[0], kin) for kin in kin_arrays):
            multipoles_in = self.power_term_multipoles(
                k=kin_arrays[0], term_list=term_list, ells=ells_tot, use_AP=use_AP
            )
        else:
            multipoles_in = {
                f"ell{ell}": self.power_term_multipoles(
                    k=kin_arrays[i], term_list=term_list, ells=[ell], use_AP=use_AP
                )[f"ell{ell}"]
                for i, ell in enumerate(ells_tot)
            }

        multipoles_out = {}
        multipoles_out["k"] = mixing_matrix.kout
        for ell in ells:
            multipoles_out[f"ell{ell}"] = sum(
                np.dot(
                    mixing_matrix.mixing[f"ELL_{ell}-{ell_prime}"],
                    multipoles_in[f"ell{ell_prime}"].T,
                ).T
                for ell_prime in ells_tot
            )

        return multipoles_out

    def _UVcutoff(self, k: np.ndarray, kcut: float, pow: float):
        r"""Cutoff of ultraviolet modes

        Parameters
        ----------
        k: np.ndarray
            Input wave modes
        kcut: float
            Cutoff scale
        pow: float
            Index of exponential cutoff

        Returns
        -------
        damping: np.ndarray
            Damping function of UV wave modes
        """
        return np.exp(-((k / kcut) ** pow))

    @format_output("2PCF_multipoles")
    def two_point_correlation_multipoles(
        self,
        s: np.ndarray,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        logkmin: Optional[float] = -6.0,
        logkmax: Optional[float] = 2.0,
        nk: Optional[int] = 2048,
        kcut: Optional[float] = 2.0,
        pow: Optional[float] = 2.0,
        format_type: Optional[str] = None,
    ) -> dict:
        r"""Two-point correlation function Legendre multipoles.

        Parameters
        ----------
        s: np.ndarray
            Comoving separations
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        logkmin: float
            Left logarithmic edge of input wave mode array
        logkmax: float
            Right logarithmic edge of input wave mode array
        nk: int
            Number of logarithmic wave mode bins
        kcut: float
            Cutoff scale for exponential damping
        pow: float
            Power index for exponential damping
        format_type: str
            Type of output format
        Returns
        -------
        multipoles: dict
            Two-point correlation function Legendre multipoles
        """
        if self.spectro_power.NLcode != "COMET":
            raise ValueError(
                "2PCF multipoles can temporarily be retrieved only with COMET"
            )

        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])
        k_hnkl = np.logspace(logkmin, logkmax, nk)
        pk_multipoles = self.power_multipoles(k=k_hnkl, ells=ells, use_AP=use_AP)
        volume_factor = (k_hnkl**3) / (2 * (np.pi**2))
        xi_multipoles = {}
        for ell in ells:
            y_array = (
                volume_factor
                * pk_multipoles[f"ell{ell}"]
                * self._UVcutoff(k=k_hnkl, kcut=kcut, pow=pow)
                * np.real(1j**ell)
            )
            transformer = fftlog(x=k_hnkl, fx=y_array, nu=2)
            r_grid, transformed_log = transformer.fftlog(ell=ell)
            xi_multipoles[f"ell{ell}"] = np.interp(s, r_grid, transformed_log)

        return xi_multipoles

    @format_output("2PCF_polar")
    def two_point_correlation_polar(
        self,
        s: np.ndarray,
        mu: np.ndarray,
        use_AP: Optional[bool] = True,
        logkmin: Optional[float] = -6.0,
        logkmax: Optional[float] = 2.0,
        nk: Optional[int] = 2048,
        kcut: Optional[float] = 2.0,
        pow: Optional[float] = 2.0,
        format_type: Optional[str] = None,
    ) -> dict:
        r"""Polar two-point correlation function.

        Parameters
        ----------
        s: np.ndarray
            Comoving separations
        mu: np.ndarray
            Cosinus of the angle between the pair separation and the line of sight
        use_AP: bool
            Flag to switch between with and without AP corrections
        logkmin: float
            Left logarithmic edge of input wave mode array
        logkmax: float
            Right logarithmic edge of input wave mode array
        nk: int
            Number of logarithmic wave mode bins
        kcut: float
            Cutoff scale for exponential damping
        pow: float
            Power index for exponential damping
        format_type: str
            Type of output format
        Returns
        -------
        xi_polar: np.ndarray
            Polar two-point correlation function
        """
        if self.spectro_power.NLcode != "COMET":
            raise ValueError("Polar 2PCF can temporarily be retrieved only with COMET")
        ells = [0, 2, 4]

        xi_multipoles = self.two_point_correlation_multipoles(
            s=s, ells=ells, use_AP=use_AP
        )

        xi_polar = sum(
            np.outer(xi_multipoles[f"ell{ell}"], legendre(ell, mu)) for ell in ells
        )

        return xi_polar

    def two_point_correlation_term_multipoles(
        self,
        s: np.ndarray,
        term_list: list,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        logkmin: Optional[float] = -6.0,
        logkmax: Optional[float] = 2.0,
        nk: Optional[int] = 2048,
        kcut: Optional[float] = 2.0,
        pow: Optional[float] = 2.0,
    ) -> dict:
        r"""Two-point correlation function Legendre multipoles of specified terms.

        Parameters
        ----------
        s: np.ndarray
            Comoving separations
        term_list: list
            List of terms to compute
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        logkmin: float
            Left logarithmic edge of input wave mode array
        logkmax: float
            Right logarithmic edge of input wave mode array
        nk: int
            Number of logarithmic wave mode bins
        kcut: float
            Cutoff scale for exponential damping
        pow: float
            Power index for exponential damping
        Returns
        -------
        multipoles: dict
            Two-point correlation function Legendre multipoles of specified terms
        """

        if self.spectro_power.NLcode != "COMET":
            raise ValueError(
                "2PCF multipoles for specific terms can temporarily be retrieved only with COMET"
            )

        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])
        k_hnkl = np.logspace(logkmin, logkmax, nk)
        pk_multipoles = self.power_term_multipoles(
            k=k_hnkl, term_list=term_list, ells=ells, use_AP=use_AP
        )
        volume_factor = (k_hnkl**3) / (2 * (np.pi**2))
        xi_multipoles = {}
        for ell in ells:
            xi_temp = np.zeros((len(term_list), len(s)))
            for term_id, term in enumerate(term_list):
                y_array = (
                    volume_factor
                    * pk_multipoles[f"ell{ell}"][term_id]
                    * self._UVcutoff(k=k_hnkl, kcut=kcut, pow=pow)
                    * np.real(1j**ell)
                )
                transformer = fftlog(x=k_hnkl, fx=y_array, nu=2)
                r_grid, transformed_log = transformer.fftlog(ell=ell)
                xi_temp[term_id, :] = np.interp(s, r_grid, transformed_log)
            xi_multipoles[f"ell{ell}"] = xi_temp
        return xi_multipoles
