"""Beyond-LCDM interface of Legendre Multipoles with PBJ."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
from typing import Optional

import numpy as np  # type: ignore

try:
    from pbjcosmo.theory import Theory
    from pbjcosmo.tools import cosmology

    pbj_obj = Theory()
except (ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"PBJ could not be imported or initialised: {e}")


class PBJSpectroPower:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PBJ."""

    NLcode = "PBJ"
    PBJ_GROWTH_MODELS = {
        "lcdm",
        "wcdm",
        "w0wacdm",
        "darkscattering",
        "ndgp",
        "growthindex",
        "fr",
        "dgp",
    }
    GROWTH_MODEL_MAP = {
        "gamma": "growthindex",
        "ide": "darkscattering",
        "wcdm": "wcdm",
        "w0wacdm": "w0wacdm",
    }
    GROWTH_MODEL_ALIASES = (
        "gravity_model",
        "growth_model",
        "growth",
        "cosmo_model",
        "model",
    )
    DGP_PARAM_ALIASES = ("omega_rc", "omegarc")
    COSMO_PARAM_ALIASES = {
        "fR0": ("fr0", "fR0"),
        "omegarc": DGP_PARAM_ALIASES,
        "gamma": ("gamma0", "gamma"),
        "xi": ("xi",),
        "Omrc": ("Omrc", *DGP_PARAM_ALIASES),
    }

    def __init__(
        self,
        linear_perturbations: Perturbations,
        nuisance_parameters: dict,
        growth_perturbations: Optional[Perturbations] = None,
        redshift: Optional[float] = None,
    ):
        r"""Class constructor.

        Args:
          linear_perturbations (Perturbations): LCDM Perturbations object used
            as the z=0 baseline linear power spectrum
          nuisance_parameters (dict): Dictionary containing bias and counterterm parameters
          growth_perturbations (Perturbations | None): Optional MGrowth
            Perturbations object used only for modified-growth f and D
          redshift (float | None): Optional prediction redshift. If omitted,
            the first nonzero redshift in the perturbation object is used.
        """
        self.linear_perturbations = linear_perturbations
        self.growth_perturbations = growth_perturbations
        self.background = linear_perturbations.background
        self.parameters = nuisance_parameters
        redshift_source = growth_perturbations or linear_perturbations
        self.z = np.asarray(redshift_source.z, dtype=float)
        if redshift is None:
            redshift_arr = self.z[self.z != 0.0]
            if redshift_arr.size == 0:
                raise ValueError("PBJSpectroPower needs at least one nonzero prediction redshift")
            self.redshift = float(redshift_arr[0])
        else:
            self.redshift = float(redshift)
        self.redshift_mask = np.isclose(self.z, self.redshift)
        self.growth_model = self._infer_growth_model()

        self.cosmo = {
            "h": self.background.h,
            "Och2": self.background.Omega_cdm0 * self.background.h**2,
            "Obh2": self.background.Omega_b0 * self.background.h**2,
            "As": self.background.As,
            "ns": self.background.ns,
            "Mnu": self.background.mnu,
            "w0": self.background.w0,
            "wa": self.background.wa,
            "Tcmb": 2.7255,
        }
        self.cosmo["Omh2"] = (
            self.cosmo["Och2"] + self.cosmo["Obh2"] + self.cosmo["Mnu"] / 93.14
        )
        self.cosmo["Om"] = self.cosmo["Omh2"] / self.cosmo["h"] ** 2
        self.cosmo.update(self._collect_beyond_lcdm_parameters())

    @staticmethod
    def _get_first_available(source, names):
        """Return the first matching attribute or mapping entry from `source`."""
        if source is None:
            return None
        for name in names:
            if hasattr(source, name):
                value = getattr(source, name)
                if value is not None:
                    return value
            if isinstance(source, dict) and name in source and source[name] is not None:
                return source[name]
        return None

    @staticmethod
    def _mgpars_from(source):
        """Return a nested MG parameter dict when a CLOE object carries one."""
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get("mgpars")
        return getattr(source, "mgpars", None)

    def _parameter_sources(self):
        """Candidate objects/dicts that may carry model or MG parameters."""
        sources = (
            self.parameters,
            self._mgpars_from(self.parameters),
            self.growth_perturbations,
            self._mgpars_from(self.growth_perturbations),
            self.linear_perturbations,
            self._mgpars_from(self.linear_perturbations),
            self.background,
        )
        return tuple(source for source in sources if source is not None)

    def _get_first_from_sources(self, sources, names):
        """Return the first matching value across multiple parameter sources."""
        for source in sources:
            value = self._get_first_available(source, names)
            if value is not None:
                return value
        return None

    def _infer_growth_model(self) -> str:
        """Infer the PBJ growth model from CLOE metadata or supplied parameters."""
        candidates = self._parameter_sources()
        for source in candidates:
            model = self._get_first_available(source, self.GROWTH_MODEL_ALIASES)
            if model is not None:
                return self._normalise_growth_model(model)

        if self._get_first_from_sources(candidates, self.COSMO_PARAM_ALIASES["fR0"]) is not None:
            return "fr"
        if self._get_first_from_sources(candidates, self.COSMO_PARAM_ALIASES["omegarc"]) is not None:
            return "dgp"
        if self._get_first_from_sources(candidates, self.COSMO_PARAM_ALIASES["Omrc"]) is not None:
            return "ndgp"
        if self._get_first_from_sources(candidates, self.COSMO_PARAM_ALIASES["xi"]) is not None:
            return "darkscattering"
        if self._get_first_from_sources(candidates, self.COSMO_PARAM_ALIASES["gamma"]) is not None:
            return "growthindex"
        return "lcdm"

    def _normalise_growth_model(self, model: str) -> str:
        """Translate CLOE model tags to PBJ growth-model tags."""
        model = str(model)
        model_key = model.lower()
        growth_model = self.GROWTH_MODEL_MAP.get(model_key, model_key)
        if growth_model not in self.PBJ_GROWTH_MODELS:
            raise ValueError(
                f"Unsupported PBJ growth model '{model}'. "
                f"Choose one of: {', '.join(sorted(self.PBJ_GROWTH_MODELS))}"
            )
        return growth_model

    def _collect_beyond_lcdm_parameters(self) -> dict:
        """Read beyond-LCDM parameters from CLOE objects when present."""
        extra_cosmo = {}
        sources = self._parameter_sources()
        for key, aliases in self.COSMO_PARAM_ALIASES.items():
            for source in sources:
                value = self._get_first_available(source, aliases)
                if value is not None:
                    extra_cosmo[key] = value
                    break
        return extra_cosmo

    @staticmethod
    def _format_growth_input(value):
        """Return scalar or 1D growth input in the shape expected by PBJ."""
        value = np.asarray(value, dtype=float)
        value = np.squeeze(value)
        if value.ndim == 0:
            return float(value)
        if value.size == 1:
            return float(np.ravel(value)[0])
        return value

    def _pbj_growth_inputs(self, kgrid: np.ndarray):
        """Return CLOE/PBJ growth inputs in PBJ-compatible normalisation."""
        karr_growth = kgrid[:, 0] if np.asarray(kgrid).ndim > 1 else kgrid

        if self.growth_model != "lcdm":
            if self.growth_model == "fr":
                growth_rate, growth_factor = cosmology.eval_growth_functions(
                    self.growth_model,
                    self.redshift,
                    karr=karr_growth / self.background.h,
                    **self.cosmo,
                )
                return (
                    self._format_growth_input(growth_rate),
                    self._format_growth_input(growth_factor),
                )

            if self.growth_perturbations is not None and hasattr(
                self.growth_perturbations, "fz_interp"
            ) and hasattr(
                self.growth_perturbations, "dz_norm_lcdm_interp"
            ):
                z_eval = np.asarray([self.redshift])
                growth_rate = self.growth_perturbations.fz_interp(z_eval, karr_growth)
                growth_factor = self.growth_perturbations.dz_norm_lcdm_interp(
                    z_eval, karr_growth
                )
                return (
                    self._format_growth_input(growth_rate),
                    self._format_growth_input(growth_factor),
                )

            return None, None

        growth_rate = np.asarray(self.linear_perturbations.growth_rate())
        if growth_rate.ndim > 0 and growth_rate.size == self.z.size:
            growth_rate = growth_rate[self.redshift_mask]
        growth_factor = np.asarray(
            self.linear_perturbations.growth_factor(self.redshift, 0.05)
        )

        return self._format_growth_input(growth_rate), self._format_growth_input(growth_factor)

    def _linear_power_at_z0(self) -> np.ndarray:
        """Return a 1D z=0 linear spectrum on PBJ's internal k grid."""
        try:
            plinear = self.linear_perturbations.matter_power_spectrum(
                0.0, pbj_obj.kL, hubble_units=False, k_hunit=False
            )
        except TypeError:
            plinear = self.linear_perturbations.matter_power_spectrum(0.0, pbj_obj.kL)

        plinear = np.asarray(plinear, dtype=float)
        plinear = np.squeeze(plinear)
        if plinear.ndim != 1:
            raise ValueError(
                "PBJ needs a 1D linear power spectrum at z=0; "
                f"got shape {plinear.shape} from matter_power_spectrum"
            )
        if plinear.size != pbj_obj.kL.size:
            raise ValueError(
                "PBJ linear power spectrum length does not match the PBJ k grid: "
                f"got {plinear.size}, expected {pbj_obj.kL.size}"
            )
        return plinear

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight

        Returns:
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        plinear = self._linear_power_at_z0()
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")
        growth_rate, growth_factor = self._pbj_growth_inputs(k)

        pkmu = pbj_obj.P_kmu_2D(
            self.redshift,
            True,
            kgrid=k,
            mu=mu,
            growth_model=self.growth_model,
            f=growth_rate,
            D=growth_factor,
            cosmo=self.cosmo,
            IRres=True,
            **self.parameters,
        )

        return pkmu

    def Pk2d_term_rsd(
        self, k: np.ndarray, mu: np.ndarray, term_list: list
    ) -> np.ndarray:
        r"""2D power spectrum for specific diagrams of the loop expansion.

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
            2D power spectrum of specific terms
        """
        plinear = self._linear_power_at_z0()
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")
        growth_rate, growth_factor = self._pbj_growth_inputs(k)

        pkmu_marg_dict = pbj_obj.P_kmu_2D_marg_dict(
            self.redshift,
            True,
            kgrid=k,
            mu=mu,
            growth_model=self.growth_model,
            f=growth_rate,
            D=growth_factor,
            cosmo=self.cosmo,
            IRres=True,
            b1=self.parameters["b1"],
        )

        Pk2d = np.array([pkmu_marg_dict[key] for key in term_list])
        return Pk2d
