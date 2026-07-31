"""Private shared implementation for CLASS-family cosmology backends.

This module intentionally does not expose a new public backend API.  Public
wrappers such as :class:`CLASSBackground` provide the solver-specific setup;
this module owns behavior that is the same for all CLASS derivatives.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Background

SolverFactory = Callable[[], Any]
Masses = Union[float, Sequence[float], np.ndarray]


class _ClassFamilyBackground:
    """Common state and background operations for CLASS-derived solvers."""

    c0 = SPEED_OF_LIGHT / 1000
    _solver_factory: SolverFactory
    _parameter_key: str

    def _initialize(
        self,
        *,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        alpha_s: float,
        mnu: Masses,
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        N_ur: Optional[float],
        solver_factory: SolverFactory,
        parameter_key: str,
        extra_parameters: Optional[dict[str, Any]] = None,
        parameter_adjuster: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.H0 = H0
        self.h = H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.alpha_s = alpha_s
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG
        self.mnu = mnu
        self.N_mnu = N_mnu
        self._provided_N_ur = N_ur
        self._solver_factory = solver_factory
        self._parameter_key = parameter_key

        self._validate_neutrinos()
        self._class_family_params: dict[str, Any] = self._common_parameters()
        if extra_parameters:
            overlap = set(extra_parameters) & set(self._class_family_params)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(
                    f"Solver parameters are controlled by CLOE and cannot be overridden: {names}"
                )
            self._class_family_params.update(extra_parameters)
        if parameter_adjuster:
            parameter_adjuster(self._class_family_params)

        # Kept as a compatibility view for callers that still inspect it.
        self.interface_args: dict[str, dict[str, Any]] = {
            self._parameter_key: self._class_family_params
        }
        self.results = self._new_solver(self._class_family_params)

    def _new_solver(self, parameters: dict[str, Any]) -> Any:
        solver = self._solver_factory()
        solver.set(parameters)
        solver.compute()
        return solver

    def _validate_neutrinos(self) -> None:
        total_mass = np.sum(self.mnu)
        if total_mass > 0 and self.N_mnu == 0:
            raise ValueError("If mnu is provided, N_mnu must be greater than 0.")
        if self.N_mnu > 0 and total_mass == 0:
            raise ValueError("If N_mnu is provided, mnu must be greater than 0.")

    def _common_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "H0": self.H0,
            "omega_b": self.Omega_b0 * self.h**2,
            "omega_cdm": self.Omega_cdm0 * self.h**2,
            "Omega_k": self.Omega_k0,
            "n_s": self.ns,
            "alpha_s": self.alpha_s,
            "A_s": self.As,
            "w0_fld": self.w0,
            "wa_fld": self.wa,
            "use_ppf": "yes",
            "Omega_Lambda": 0.0,
            "N_ncdm": self.N_mnu,
            "N_ur": self.N_ur,
        }
        if self.N_mnu > 0:
            parameters["m_ncdm"] = self._set_neutrino_masses()
        return parameters

    @property
    def _interface_args(self) -> dict:
        return self.interface_args

    @property
    def N_ur(self) -> float:
        if self._provided_N_ur is not None:
            return self._provided_N_ur
        values = {0: 3.044, 1: 2.0308, 2: 1.0176, 3: 0.0044}
        try:
            return values[self.N_mnu]
        except KeyError as error:
            raise ValueError(
                "N_ur can only be inferred for 0, 1, 2, or 3 massive neutrino species."
            ) from error

    @property
    def N_eff(self) -> float:
        return self.results.Neff()

    def _set_neutrino_masses(self) -> str:
        if isinstance(self.mnu, (float, np.floating)):
            if self.N_mnu == 1:
                return f"{self.mnu:g}"
            if self.N_mnu > 1:
                return ",".join(f"{self.mnu / self.N_mnu:g}" for _ in range(self.N_mnu))
        elif isinstance(self.mnu, (np.ndarray, Sequence)):
            if len(self.mnu) != self.N_mnu:
                raise ValueError(
                    f"Expected {self.N_mnu} individual neutrino masses, but got {len(self.mnu)}: {self.mnu}"
                )
            return ",".join(f"{mass:g}" for mass in self.mnu)
        raise TypeError("mnu must be a float, numpy.ndarray or Sequence of floats")

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        H = np.array([self.results.Hubble(z) for z in zs])
        if units == "km/s/Mpc":
            return H * self.c0
        if units == "1/Mpc":
            return H
        raise ValueError("Unsupported units. Must be 'km/s/Mpc' or '1/Mpc'.")

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        return np.array([self.results.comoving_distance(z) for z in zs])

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        chi = self.comoving_distance(zs)
        if self.Omega_k0 == 0.0:
            return chi

        hubble_distance = self.c0 / self.H0
        if self.Omega_k0 > 0.0:
            sqrt_curvature = np.sqrt(self.Omega_k0)
            return (
                hubble_distance
                / sqrt_curvature
                * np.sinh(sqrt_curvature * chi / hubble_distance)
            )

        sqrt_curvature = np.sqrt(-self.Omega_k0)
        return (
            hubble_distance
            / sqrt_curvature
            * np.sin(sqrt_curvature * chi / hubble_distance)
        )

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        return np.array([self.results.angular_distance(z) for z in zs])

    def Omega_cb(self, zs: np.ndarray) -> np.ndarray:
        return self.results.Om_b(zs) + self.results.Om_cdm(zs)

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        return self.results.Om_m(zs)

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        return self.results.Om_b(zs)

    @property
    def rdrag(self) -> float:
        return self.results.rs_drag()

    @property
    def z_star(self) -> float:
        return self.results.get_current_derived_parameters(["z_star"])["z_star"]


class _ClassFamilyPerturbations:
    """Shared perturbation implementation for CLASS-derived solvers."""

    _solver_factory: SolverFactory
    _parameter_key: str
    _scale_dependent_growth = False

    def _initialize(
        self,
        background: Background,
        redshifts: np.ndarray,
        *,
        solver_factory: SolverFactory,
        parameter_key: str,
        nonlinear_model: str,
        scale_dependent_growth: bool,
        hmcode_version: Optional[str] = None,
    ) -> None:
        if not isinstance(background, _ClassFamilyBackground):
            raise TypeError(
                "CLASS-family perturbations require a background from the same CLASS-family backend."
            )
        self.background = background
        self.z = np.asarray(redshifts)
        self.kmax = 100
        self._solver_factory = solver_factory
        self._parameter_key = parameter_key
        self._scale_dependent_growth = scale_dependent_growth

        self._class_family_params = copy.deepcopy(background._class_family_params)
        self._class_family_params.update(
            {
                "output": "mPk, mTk",
                "P_k_max_1/Mpc": self.kmax,
                "k_per_decade_for_bao": 70,
                "k_per_decade_for_pk": 10,
                "z_max_pk": np.max(self.z),
                "non linear": nonlinear_model,
            }
        )
        if nonlinear_model != "none":
            self._class_family_params.update(
                {"nonlinear_min_k_max": 50, "hmcode_tol_sigma": 1e-8}
            )
        if hmcode_version is not None:
            self._class_family_params["hmcode_version"] = hmcode_version

        self.interface_args = {self._parameter_key: self._class_family_params}
        self.results = self._new_solver(self._class_family_params)
        self.k = np.logspace(np.log10(1e-4), np.log10(self.kmax), 100)
        self._linear_results: Any = None

    def _new_solver(self, parameters: dict[str, Any]) -> Any:
        solver = self._solver_factory()
        solver.set(parameters)
        solver.compute()
        return solver

    @property
    def _interface_args(self) -> dict:
        return self.interface_args

    def matter_power_spectrum(
        self,
        zs: np.ndarray,
        ks: np.ndarray,
        hubble_units: bool = False,
        k_hunit: bool = False,
    ) -> np.ndarray:
        if hubble_units or k_hunit:
            raise ValueError("CLASS-family backends do not support h-units.")
        return np.array([[self.results.pk(k, z) for k in ks] for z in zs])

    def matter_power_spectrum_cb(
        self,
        zs: np.ndarray,
        ks: np.ndarray,
        hubble_units: bool = False,
        k_hunit: bool = False,
    ) -> np.ndarray:
        if hubble_units or k_hunit:
            raise ValueError("CLASS-family backends do not support h-units.")
        if self._class_family_params["N_ncdm"] == 0:
            warnings.warn(
                "There are no massive neutrinos (N_mnu=0); returning the total matter power spectrum.",
                UserWarning,
                stacklevel=2,
            )
            return self.matter_power_spectrum(zs, ks)
        return np.array([[self.results.pk_cb(k, z) for k in ks] for z in zs])

    def _linear_solver(self) -> Any:
        if self._linear_results is None:
            parameters = copy.deepcopy(self._class_family_params)
            parameters["non linear"] = "none"
            self._linear_results = self._new_solver(parameters)
        return self._linear_results

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Return linear growth, including for nonlinear perturbation objects."""
        linear_results = self._linear_solver()
        return np.sqrt(
            np.array([[linear_results.pk(k, z) for k in ks] for z in zs])
            / np.array([[linear_results.pk(k, 0.0) for k in ks] for _ in zs])
        )

    def growth_rate(
        self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        redshifts = self.z if zs is None else np.asarray(zs)
        if self._scale_dependent_growth:
            if ks is None:
                raise ValueError(
                    "Scale-dependent growth requires explicit ks in 1/Mpc."
                )
            wavenumbers = np.asarray(ks)
            return np.array(
                [
                    [
                        self.results.scale_dependent_growth_factor_f(k, z)
                        for k in wavenumbers
                    ]
                    for z in redshifts
                ]
            )
        return np.array(
            [self.results.scale_independent_growth_factor_f(z) for z in redshifts]
        )

    def sigma8_0(self) -> float:
        return self.results.sigma8()
