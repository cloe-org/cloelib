"""CLASS cosmology wrappers built on the shared CLASS-family implementation."""

from typing import Optional, Sequence, Union

import numpy as np

from cloelib.cosmology._class_family import (
    _ClassFamilyBackground,
    _ClassFamilyPerturbations,
)
from cloelib.cosmology.cosmology import Background

try:
    from classy import Class  # type: ignore
except ImportError as error:
    raise ImportError("classy could not be imported.") from error


class CLASSBackground(_ClassFamilyBackground):
    """A wrapper for CLASS background cosmological calculations."""

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
        **kwargs,
    ) -> None:
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
            solver_factory=Class,
            parameter_key="CLASSparams",
        )


class CLASSLinearPerturbations(_ClassFamilyPerturbations):
    """Linear CLASS perturbations."""

    def __init__(self, background: Background, redshifts: np.ndarray):
        self._initialize(
            background,
            redshifts,
            solver_factory=Class,
            parameter_key="CLASSparams",
            nonlinear_model="none",
            scale_dependent_growth=False,
        )


class CLASSNonLinearPerturbations(_ClassFamilyPerturbations):
    """Nonlinear CLASS perturbations."""

    def __init__(
        self,
        background: Background,
        linearperturbations: Optional[object],
        redshifts: np.ndarray,
        nonlinear_model: Optional[str] = None,
        hmcode_version: Optional[str] = None,
    ) -> None:
        self._initialize(
            background,
            redshifts,
            solver_factory=Class,
            parameter_key="CLASSparams",
            nonlinear_model="none" if nonlinear_model is None else nonlinear_model,
            scale_dependent_growth=False,
            hmcode_version=hmcode_version,
        )
