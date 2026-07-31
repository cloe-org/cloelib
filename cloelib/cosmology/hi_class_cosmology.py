"""hi_class cosmology wrappers built on the shared CLASS-family implementation."""

from typing import Any, Optional, Sequence, Union

import numpy as np

from cloelib.cosmology._class_family import (
    _ClassFamilyBackground,
    _ClassFamilyPerturbations,
)
from cloelib.cosmology.cosmology import Background

try:
    from hiclassy import HiClass  # type: ignore
except ImportError as error:
    raise ImportError("hiclassy could not be imported.") from error


class hi_classBackground(_ClassFamilyBackground):
    """A wrapper for hi_class background cosmological calculations."""

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu: Union[float, Sequence[float], np.ndarray],
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        N_ur: Optional[float] = None,
        alpha_s: float = 0.0,
        params_smg: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        params_smg = {} if params_smg is None else dict(params_smg)
        common_keys = {
            "H0",
            "omega_b",
            "omega_cdm",
            "Omega_k",
            "n_s",
            "alpha_s",
            "A_s",
            "w0_fld",
            "wa_fld",
            "use_ppf",
            "Omega_Lambda",
            "N_ncdm",
            "N_ur",
            "m_ncdm",
        }
        overlap = set(params_smg) & common_keys
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"'{names}' is passed as an argument or has an enforced default value and cannot be in params_smg."
            )

        should_remove_fluid_parameters = (
            "gravity_model" in params_smg
            and float(params_smg.get("Omega_fld", 0.0)) == 0.0
        )
        extra_parameters = dict(params_smg)
        if "gravity_model" in params_smg:
            extra_parameters["Omega_fld"] = params_smg.get("Omega_fld", 0.0)

        def adjust_fluid_parameters(parameters: dict[str, Any]) -> None:
            if should_remove_fluid_parameters:
                for name in ("w0_fld", "wa_fld", "use_ppf"):
                    parameters.pop(name)

        self._initialize(
            H0=H0,
            Omega_b0=Omega_b0,
            Omega_cdm0=Omega_cdm0,
            Omega_k0=Omega_k0,
            As=As,
            ns=ns,
            alpha_s=alpha_s,
            mnu=mnu,
            w0=w0,
            wa=wa,
            gamma_MG=gamma_MG,
            N_mnu=N_mnu,
            N_ur=N_ur,
            solver_factory=HiClass,
            parameter_key="hi_classparams",
            extra_parameters=extra_parameters,
            parameter_adjuster=adjust_fluid_parameters,
        )

    def get_background(self) -> dict:
        """Return all hi_class background quantities."""
        return self.results.get_background()


class hi_classLinearPerturbations(_ClassFamilyPerturbations):
    """Linear hi_class perturbations."""

    def __init__(self, background: Background, redshifts: np.ndarray):
        self._initialize(
            background,
            redshifts,
            solver_factory=HiClass,
            parameter_key="hi_classparams",
            nonlinear_model="none",
            scale_dependent_growth=True,
        )


class hi_classNonLinearPerturbations(_ClassFamilyPerturbations):
    """Nonlinear hi_class perturbations."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Optional[object],
        redshifts: np.ndarray,
        nonlinear_model: Optional[str] = None,
    ) -> None:
        self._initialize(
            background,
            redshifts,
            solver_factory=HiClass,
            parameter_key="hi_classparams",
            nonlinear_model="none" if nonlinear_model is None else nonlinear_model,
            scale_dependent_growth=True,
        )
