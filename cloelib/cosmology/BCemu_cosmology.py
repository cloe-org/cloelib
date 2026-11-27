"""
Baryonic correction of the matter power spectrum from BCemu (Giri & Schneider 2021).

## Design

**``BCemuBaryonBoostMixin``** -- concrete subclass of ``BaryonBoostMixin``
(defined in ``cosmology.py``). Provides ``baryonic_suppression`` from the BCemu
baryonification emulators. Backend-agnostic: compose it with *any* nonlinear
perturbations class through
:func:`~cloelib.cosmology.cosmology.with_baryon_boost`.

**``CAMBNonLinearBCemuBaryonicPerturbations``** -- convenience alias of CAMB
nonlinear perturbations with the BCemu boost, as done for FLAMINGO.

## Models

BCemu ships three emulators, selected by the parameters passed:

=========  =======================================================  ==========
model      free baryonification parameters                          fixed by BCemu
=========  =======================================================  ==========
1-param    ``log10Mc``                                              the other six
3-param    ``log10Mc, thej, deta``                                  mu, gamma, delta, eta
7-param    ``log10Mc, mu, thej, gamma, delta, eta, deta``           --
=========  =======================================================  ==========

Each parameter may evolve with redshift as ``p(z) = p (1 + z)^(-nu_p)`` (BCemu's
native ``nu_Mc, nu_mu, nu_thej, nu_gamma, nu_delta, nu_eta, nu_deta``; all zero,
i.e. redshift independent, by default). The baryon fraction ``fb = Omega_b /
Omega_m`` is taken from the background.

## Ranges

BCemu is trained at z = 0, 0.5, 1, 1.5, 2 and 0.034 <= k <= 12.5 h/Mpc:

- between those redshifts the boost is interpolated linearly in z; above z = 2
  it is held at its z = 2 value (``z_extrapolation="const"``) or extended
  linearly from the last two nodes (``"linear"``);
- below the trained k range B -> 1 (B = 1 at k <= 0.01 h/Mpc, joined in log k);
- above it the boost is held at its last value (``k_extrapolation="const"``) or
  extended with a power law in k (``"power_law"``).

Parameters outside the training box, including a derived ``fb`` outside
[0.10, 0.25], raise a ``ValueError`` (BCemu itself raises a bare
``AssertionError``).

## Notes
- The BCemu emulators are expensive to load, so one instance per model is
  created on first use and reused; the per-call state BCemu keeps on the
  instance (parameters, redshift evolution, ``fb``) is reset on every call.
"""

# General imports
import numpy as np
from scipy import interpolate
from typing import Optional

# Cosmology imports
from cloelib.cosmology.cosmology import (
    Background,
    BaryonBoostMixin,
    with_baryon_boost,
)
from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations

try:
    import BCemu
    from BCemu.BaryonEffectsEmulator import ks_emulated as _BCEMU_K
except ImportError:
    raise ImportError("BCemu could not be imported.")


#: Redshifts at which the BCemu emulators are trained.
_Z_NODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
#: Trained wavenumber range in h/Mpc.
_K_MIN_TRAINED = float(np.min(_BCEMU_K))
_K_MAX_TRAINED = float(np.max(_BCEMU_K))
#: Below this wavenumber (h/Mpc) the suppression is set to exactly 1.
_K_UNITY = 0.01

#: model name -> (BCemu class, free parameters in BCemu's own order)
_MODELS = {
    1: ("BCM_1param", ("log10Mc",)),
    3: ("BCM_3param", ("log10Mc", "thej", "deta")),
    7: ("BCM_7param", ("log10Mc", "mu", "thej", "gamma", "delta", "eta", "deta")),
}
_NU_NAMES = ("nu_Mc", "nu_mu", "nu_thej", "nu_gamma", "nu_delta", "nu_eta", "nu_deta")

#: One BCemu instance per model (loading unpickles five trained emulators).
_EMULATORS: dict = {}


def _bcemu_instance(n_params: int):
    """Return the cached BCemu emulator for the 1-, 3- or 7-parameter model."""
    if n_params not in _EMULATORS:
        _EMULATORS[n_params] = getattr(BCemu, _MODELS[n_params][0])(verbose=False)
    return _EMULATORS[n_params]


class BCemuBaryonBoostMixin(BaryonBoostMixin):
    """Concrete baryonic-boost mixin for the BCemu emulator.

    Combine it with any nonlinear perturbations class::

        from cloelib.cosmology.cosmology import with_baryon_boost
        from cloelib.cosmology.BCemu_cosmology import BCemuBaryonBoostMixin

        MyPert = with_baryon_boost(SomeNonLinearPerturbations, BCemuBaryonBoostMixin)
        pert = MyPert(
            background, linear_pert, redshifts,
            baryon_kwargs=dict(log10Mc=13.3, thej=4.2, deta=0.14),
        )

    The model (1, 3 or 7 parameters) is inferred from the baryonification
    parameters given, or set explicitly with ``n_params``.

    .. note::
        ``__init__`` must be called **after** the base NonLinear ``__init__``
        because it reads ``self.background``.
    """

    #: Supplied by the nonlinear perturbations class this mixin is composed with.
    background: Background

    def __init__(
        self,
        log10Mc: float,
        thej: Optional[float] = None,
        deta: Optional[float] = None,
        mu: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        eta: Optional[float] = None,
        nu_Mc: float = 0.0,
        nu_mu: float = 0.0,
        nu_thej: float = 0.0,
        nu_gamma: float = 0.0,
        nu_delta: float = 0.0,
        nu_eta: float = 0.0,
        nu_deta: float = 0.0,
        n_params: Optional[int] = None,
        k_extrapolation: str = "const",
        z_extrapolation: str = "const",
    ) -> None:
        """Initialise the BCemu boost table.

        The baryonification parameters carry **no default values**: every
        parameter of the chosen model must be given, so that no arbitrary
        feedback scenario stands in for a real choice. The redshift-evolution
        parameters default to 0, BCemu's own redshift-independent case.

        Parameters
        ----------
        log10Mc, thej, deta, mu, gamma, delta, eta:
            Baryonification parameters (BCemu names and units). Give
            ``log10Mc`` alone (1-param model), with ``thej`` and ``deta``
            (3-param model), or all seven (7-param model).
        nu_Mc, nu_mu, nu_thej, nu_gamma, nu_delta, nu_eta, nu_deta:
            Redshift evolution, ``p(z) = p (1 + z)^(-nu_p)``. Only those of the
            free parameters of the chosen model are meaningful.
        n_params:
            Force the 1-, 3- or 7-parameter model instead of inferring it.
        k_extrapolation:
            Above the trained k range: ``"const"`` (hold the last value) or
            ``"power_law"``.
        z_extrapolation:
            Above z = 2: ``"const"`` (hold the z = 2 value) or ``"linear"``
            (extend linearly in z from the last two nodes).

        Raises
        ------
        ValueError
            If the parameters do not match a BCemu model, if an extrapolation
            option is unknown, or if a parameter (or ``fb = Omega_b/Omega_m``)
            is outside the BCemu training box.
        """
        given = {
            name: value
            for name, value in dict(
                log10Mc=log10Mc,
                mu=mu,
                thej=thej,
                gamma=gamma,
                delta=delta,
                eta=eta,
                deta=deta,
            ).items()
            if value is not None
        }
        if n_params is None:
            n_params = {1: 1, 3: 3, 7: 7}.get(len(given), 0)
        if n_params not in _MODELS or set(given) != set(_MODELS[n_params][1]):
            raise ValueError(
                "BCemu needs the parameters of exactly one of its models: "
                + "; ".join(
                    f"{n}-param: {', '.join(p)}" for n, (_, p) in _MODELS.items()
                )
                + f". Got: {', '.join(sorted(given)) or 'none'}"
                + (f" with n_params={n_params}." if n_params else ".")
            )
        if k_extrapolation not in ("const", "power_law"):
            raise ValueError(f"Unknown k_extrapolation={k_extrapolation!r}.")
        if z_extrapolation not in ("const", "linear"):
            raise ValueError(f"Unknown z_extrapolation={z_extrapolation!r}.")

        self.n_params = n_params
        self.bcemu_params = {name: float(given[name]) for name in _MODELS[n_params][1]}
        self.bcemu_nu = dict(
            zip(
                _NU_NAMES,
                map(
                    float, (nu_Mc, nu_mu, nu_thej, nu_gamma, nu_delta, nu_eta, nu_deta)
                ),
            )
        )
        self.k_extrapolation = k_extrapolation
        self.z_extrapolation = z_extrapolation
        self.fb = float(self.background.Omega_b0 / self.background.Omega_m(0.0))

        k_h = np.asarray(_BCEMU_K, dtype=float)
        S = self._emulate(k_h)  # (n_z_nodes, n_k)

        # Join B = 1 at k <= _K_UNITY to the first trained wavenumber.
        self._lnk_nodes = np.log(np.concatenate([[_K_UNITY], k_h]))
        lnS = np.log(np.concatenate([np.ones((len(_Z_NODES), 1)), S], axis=1))
        self._lnS_interp = interpolate.RectBivariateSpline(
            _Z_NODES, self._lnk_nodes, lnS, kx=1, ky=1
        )
        # Logarithmic slope at the last trained wavenumber, for "power_law".
        self._high_k_slope = (lnS[:, -1] - lnS[:, -2]) / (
            self._lnk_nodes[-1] - self._lnk_nodes[-2]
        )
        self._lnS_kmax = lnS[:, -1]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _emulate(self, k_h: np.ndarray) -> np.ndarray:
        """S(z_node, k) at the BCemu training redshifts, k in h/Mpc."""
        emulator = _bcemu_instance(self.n_params)
        try:
            # BCemu updates its instance with the dict it is given, so every
            # parameter and every redshift-evolution term is passed explicitly
            # on each call: anything omitted would keep the previous call's value.
            return np.array(
                [
                    emulator.get_boost(
                        float(z),
                        {**self.bcemu_params, **self.bcemu_nu},
                        k_h,
                        fb=self.fb,
                    )
                    for z in _Z_NODES
                ]
            )
        except AssertionError as err:
            raise ValueError(
                f"BCemu parameters outside the training box: {self.bcemu_params}, "
                f"{self.bcemu_nu}, fb = Omega_b/Omega_m = {self.fb:.4f} "
                "(allowed 0.10-0.25). " + str(err).strip()
            ) from err

    @property
    def baryon_k_range(self) -> tuple[float, float]:
        """Wavenumbers in 1/Mpc over which ``baryonic_suppression`` is emulated."""
        return (_K_MIN_TRAINED * self.background.h, _K_MAX_TRAINED * self.background.h)

    def _k_to_hMpc(self, ks: np.ndarray, k_hunit: bool) -> np.ndarray:
        """Return ks in h/Mpc (convert from 1/Mpc when k_hunit is False)."""
        return ks if k_hunit else ks / self.background.h

    # ------------------------------------------------------------------
    # BaryonBoostMixin implementation
    # ------------------------------------------------------------------

    def baryonic_suppression(
        self, zs: np.ndarray, ks: np.ndarray, k_hunit: bool = False
    ) -> np.ndarray:
        """Return the BCemu baryonic suppression factor P_hydro/P_DMO.

        Parameters
        ----------
        zs:
            Redshifts, shape (nz,).
        ks:
            Wavenumbers, shape (nk,). In 1/Mpc unless ``k_hunit`` is True.
        k_hunit:
            If ``True`` ks are in h/Mpc; otherwise in 1/Mpc.

        Returns
        -------
        np.ndarray, shape (nz, nk)
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        lnk = np.log(self._k_to_hMpc(ks, k_hunit))
        lnk_c = np.clip(lnk, self._lnk_nodes[0], self._lnk_nodes[-1])

        z_top = _Z_NODES[-1]
        zc = np.clip(zs, 0.0, z_top)
        lnS = self._lnS_interp(zc, lnk_c)
        if self.z_extrapolation == "linear" and np.any(zs > z_top):
            dz = _Z_NODES[-1] - _Z_NODES[-2]
            lnS_top = self._lnS_interp([z_top], lnk_c)[0]
            slope = lnS_top - self._lnS_interp([z_top - dz], lnk_c)[0]
            above = zs > z_top
            lnS[above] = lnS_top + slope * ((zs[above] - z_top) / dz)[:, None]

        if self.k_extrapolation == "power_law":
            high = lnk > self._lnk_nodes[-1]
            if np.any(high):
                slope = np.interp(zc, _Z_NODES, self._high_k_slope)
                lnS[:, high] += slope[:, None] * (lnk[high] - self._lnk_nodes[-1])
        return np.exp(lnS)


#: CAMB nonlinear perturbations with the BCemu baryonic correction.
#:
#: Created via :func:`~cloelib.cosmology.cosmology.with_baryon_boost`.
#: Accepts all ``CAMBNonLinearPerturbations`` arguments plus a
#: ``baryon_kwargs`` dict forwarded to ``BCemuBaryonBoostMixin``::
#:
#:     pert = CAMBNonLinearBCemuBaryonicPerturbations(
#:         background=bg,
#:         linearperturbations=lin_pert,
#:         redshifts=zs,
#:         baryon_kwargs=dict(log10Mc=13.32, thej=4.235, deta=0.14),
#:     )
CAMBNonLinearBCemuBaryonicPerturbations = with_baryon_boost(
    CAMBNonLinearPerturbations, BCemuBaryonBoostMixin
)
