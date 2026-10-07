"""Interface of Legendre Multipoles with PyBird."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
import warnings
from typing import Optional

import numpy as np  # type: ignore
from scipy.interpolate import make_interp_spline
from scipy.special import legendre

# Largest wavenumber [h/Mpc] at which PyBird is evaluated. Beyond it the
# multipoles are extrapolated, so it must cover the largest k requested
# downstream, e.g. the k_in range of the mixing matrices (up to ~1 h/Mpc for
# Euclid DR1). It must stay below 1 h/Mpc, the edge of the linear power
# spectrum PyBird tabulates when it runs CLASS for beyond-LCDM models.
KMAX = 0.95

PYBIRD_CONFIG = {
    "output": "bPk",
    "multipole": 3,
    "kmax": KMAX,
    "km": 1.0,
    "kr": 1.0,
    "eft_basis": "pbj",
    "with_bias": False,
    "with_stoch": False,
    "with_resum": True,
    "optiresum": True,
    "with_nnlo_counterterm": True,
}

# Analytically-marginalisable terms: likelihood (PBJ) name -> PyBird name
MARG_TERMS = {"bG3": "bGamma3", "c0": "c0", "c2": "c2", "c4": "c4", "ck4": "ct"}

ells = [0, 2, 4]

try:
    from pybird.correlator import Correlator

    N = Correlator()
    N.set(dict(PYBIRD_CONFIG))
except (ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"PyBird could not be imported or initialised: {e}")

# Configuration the shared Correlator is currently set with
_current_config = dict(PYBIRD_CONFIG)


def _set_correlator(config: dict, force: bool = False):
    """Replace the shared Correlator if its configuration changes."""
    global N, _current_config
    if force or config != _current_config:
        # A fresh Correlator: Correlator.set keeps the options of earlier calls
        N = Correlator()
        N.set(dict(config))
        _current_config = dict(config)


class PyBirdSpectroPower:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PyBird"""

    NLcode = "PyBird"

    def __init__(
        self,
        linear_perturbations: Perturbations,
        nuisance_parameters: dict,
        redshift: float,
        mg_settings: Optional[dict] = None,
    ):
        r"""Class constructor.

        Args:
          linear_perturbations (Perturbations): Perturbations object containing cosmology, linear power spectrum,
            redshift and growth functions
          nuisance_parameters (dict): Dictionary containing bias and counterterm parameters
          redshift (float): single redshift in which to evaluate PyBird
          mg_settings (dict): Optional beyond-LCDM model, passed as-is to PyBird: ``"mg_model"`` (PyBird model
            name, e.g. ``"nDGP"``) plus the PyBird options of that model, e.g.
            ``{"mg_model": "nDGP", "logOmegarc": -1.0}``. PyBird then computes the linear power spectrum and
            growth with CLASS and exact time dependence, instead of taking them from ``linear_perturbations``.
        """
        self.linear_perturbations = linear_perturbations
        self.background = linear_perturbations.background
        self.parameters = nuisance_parameters
        self.mg_settings = mg_settings

        assert np.asarray(redshift).size == 1, "Only a single redshift can be passed."
        assert redshift in linear_perturbations.z, (
            "Redshift requested for PyBird not previously computed with linear theory code"
        )
        self.redshift = redshift
        z = self.redshift
        h = self.background.h

        if mg_settings is None:
            _set_correlator(PYBIRD_CONFIG)
            kk = np.geomspace(1.0e-4, 2.0, 512)
            plin = self.linear_perturbations.matter_power_spectrum(
                z, kk, hubble_units=True, k_hunit=True
            )
            f = self.linear_perturbations.growth_rate()[
                self.linear_perturbations.z == z
            ][0]
            # No growth factor: with_time=True, PyBird takes P_lin at z directly
            N.compute({"kk": kk, "pk_lin": plin, "z": z, "f": f})
        else:
            if "mg_model" not in mg_settings:
                raise KeyError("mg_settings must contain 'mg_model'.")
            if self.background.mnu > 0:
                raise NotImplementedError(
                    "Massive neutrinos are not passed to CLASS in the beyond-LCDM "
                    "PyBird route."
                )
            # Always re-set: the Correlator keeps the redshift of its last CLASS call
            _set_correlator(
                {**PYBIRD_CONFIG, "with_exact_time": True, **mg_settings}, force=True
            )
            N.compute(self._class_parameters(z), "class")

        k_ = N.co.k  # [h/Mpc]
        self.kmax = k_[-1] * h  # [1/Mpc]

        nuisance_parameters_pybird = self.rename_nuisance_parameters(self.parameters)
        for c in ["c0", "c2", "c4"]:
            nuisance_parameters_pybird[c] *= h**2  # [Mpc]^2 -> [Mpc/h]^2
        for c in ["ct"]:
            nuisance_parameters_pybird[c] *= (
                -(h**4)
            )  # [Mpc]^4 -> [Mpc/h]^4 + sign sitch to match PBJ convention

        pkl = N.get(
            nuisance_parameters_pybird
        )  # shape (3, Nk) # k: [h/Mpc], Pk: [Mpc/h]^3
        self.ipkl = make_interp_spline(
            k_ * h, pkl / h**3, k=3, axis=-1
        )  # k: [1/Mpc], Pk: [Mpc]^3

        self.marg_parameter_names = list(MARG_TERMS.values())

        pkl_term = N.getmarg(
            nuisance_parameters_pybird,
            self.marg_parameter_names,
        ).reshape(-1, 3, k_.shape[0])  # shape (Nterm, Nk*Nl) -> (Nterm, Nl, Nk)

        for i, c in enumerate(self.marg_parameter_names):
            if c in ["c0", "c2", "c4"]:
                pkl_term[i] *= h**2  # [Mpc]^2 -> [Mpc/h]^2
            if c in ["ct"]:
                pkl_term[i] *= (
                    -(h**4)
                )  # [Mpc]^4 -> [Mpc/h]^4 + sign sitch to match PBJ convention

        self.ipkl_term = make_interp_spline(
            k_ * h, pkl_term / h**3, k=3, axis=-1
        )  # k: [1/Mpc], Pk: [Mpc]^3

    def _class_parameters(self, z: float) -> dict:
        """CLASS input parameters of the background cosmology, at redshift z."""
        bg = self.background
        return {
            "h": bg.h,
            "Omega_b": bg.Omega_b0,
            "Omega_cdm": bg.Omega_cdm0,
            "Omega_k": bg.Omega_k0,
            "A_s": bg.As,
            "n_s": bg.ns,
            "alpha_s": getattr(bg, "alpha_s", 0.0),
            "w0_fld": bg.w0,
            "wa_fld": bg.wa,
            "Omega_Lambda": 0.0,
            "z": z,
        }

    def _check_k_range(self, k: np.ndarray):
        """Warn if k goes well beyond the range PyBird was evaluated on."""
        if np.max(k) > 1.2 * self.kmax:
            warnings.warn(
                "PyBird multipoles are extrapolated well beyond the k range they "
                "were computed on; increase KMAX in cloelib.observables.PyBird_spectro."
            )

    def rename_nuisance_parameters(self, params):
        """PBJ/Comet to PyBird convention"""
        keys = set(params)

        ct_name = "cnlo" if "cnlo" in keys else "ck4" if "ck4" in keys else None

        bGamma3_name = "bGam3" if "bGam3" in keys else "bG3" if "bG3" in keys else None

        if ct_name is None:
            raise KeyError(
                "Could not detect whether 'ct' should map to 'cnlo' or 'ck4'."
            )
        if bGamma3_name is None:
            raise KeyError(
                "Could not detect whether 'bGamma3' should map to 'bGam3' or 'bG3'."
            )

        key_map = {
            "b1": "b1",
            "bt2": "b2",
            "bG2": "bG2",
            "bGamma3": bGamma3_name,
            "c0": "c0",
            "c2": "c2",
            "c4": "c4",
            "ct": ct_name,
        }

        return {new_key: params[old_key] for new_key, old_key in key_map.items()}

    @staticmethod
    def _sumrule(k: np.ndarray, mu: np.ndarray, prefix: str = "") -> str:
        """Einsum rule projecting multipoles on mu (1D k, or 2D k mesh with AP)."""
        if k.ndim == 1:
            return f"{prefix}lk,lm->{prefix}km"
        if k.ndim == 2 and k.shape[1] == mu.shape[0]:
            return f"{prefix}lkm,lm->{prefix}km"
        raise ValueError(
            f"k must be 1D or a 2D mesh matching mu; got shapes {k.shape}, {mu.shape}."
        )

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight

        Returns:
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        self._check_k_range(k)
        leglmu = np.array([legendre(ell)(mu) for ell in ells])
        return np.einsum(self._sumrule(k, mu), self.ipkl(k), leglmu)

    def Pk2d_term_rsd(
        self, k: np.ndarray, mu: np.ndarray, term_list: list
    ) -> np.ndarray:
        r"""2D power spectrum for a subset of specific diagrams of the loop expansion,
        corresponding to linear parameters that can be analytically marginalised over.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        term_list: list
            Identifiers of loop diagrams.
            Available options: 'bG3', 'c0', 'c2', 'c4', 'ck4'

        Returns
        -------
        Pk2d: np.ndarray
            2D power spectrum of specific terms, in the order of ``term_list``
        """
        unknown = [term for term in term_list if term not in MARG_TERMS]
        if unknown:
            raise ValueError(
                f"Unknown terms {unknown}; available options: {list(MARG_TERMS)}."
            )
        self._check_k_range(k)
        idx = [list(MARG_TERMS).index(term) for term in term_list]
        leglmu = np.array([legendre(ell)(mu) for ell in ells])
        return np.einsum(
            self._sumrule(k, mu, prefix="f"), self.ipkl_term(k)[idx], leglmu
        )
