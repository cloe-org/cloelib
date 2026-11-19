"""Interface of Legendre Multipoles with PBJ."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
from typing import Protocol
import numpy as np

try:
    from pbjcosmo.theory import Theory

    pbj_obj = Theory()
except (ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"PBJ could not be imported or initialised: {e}")


class PBJLinearPerturbations(Perturbations, Protocol):
    """The `Perturbations` interface plus the extras `PBJSpectroPower` uses.

    `.z`, `growth_factor_cb` and the `hubble_units`/`k_hunit` keywords of
    `matter_power_spectrum(_cb)` are not part of `Perturbations` (not every
    backend provides them), but the Boltzmann-code backends PBJ is used
    with (CAMB, CLASS and variants) do.
    """

    @property
    def z(self) -> np.ndarray:
        """Redshift grid the linear quantities were computed on."""
        ...

    def matter_power_spectrum(
        self, zs, ks, hubble_units: bool = False, k_hunit: bool = False
    ) -> np.ndarray:
        """Linear total-matter power spectrum."""
        ...

    def matter_power_spectrum_cb(
        self, zs, ks, hubble_units: bool = False, k_hunit: bool = False
    ) -> np.ndarray:
        """Linear cdm+baryon power spectrum."""
        ...

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Growth factor, also for a scalar redshift/wavenumber."""
        ...

    def growth_factor_cb(self, zs, ks) -> np.ndarray:
        """Growth factor of cdm+baryons."""
        ...


class PBJSpectroPowerDarkEnergy:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PBJ."""

    NLcode = "PBJ"

    def __init__(
        self,
        linear_perturbations_lcdm: PBJLinearPerturbations,
        linear_perturbations_darkenergy: PBJLinearPerturbations,
        nuisance_parameters: dict,
        redshift: float,
    ):
        r"""Class constructor.

        The PBJ loop kernels are built from the LCDM linear spectrum in
        1/Mpc. Growth is taken from the dark-energy perturbations, with
        :math:`D = D_{\rm de}(z) / D_{\Lambda{\rm CDM}}(0)`. ``background`` is
        the dark-energy background so a later Alcock-Paczynski step can read
        its distances. The dictionary passed to PBJ stays LCDM (``w0=-1``,
        ``wa=0``) and must share :math:`\Omega`, :math:`h`, :math:`A_s` and
        :math:`n_s` with the LCDM spectrum.

        Args:
          linear_perturbations_lcdm (PBJLinearPerturbations): LCDM linear
            spectrum used for the PBJ kernels.
          linear_perturbations_darkenergy (PBJLinearPerturbations): Binned-w
            perturbations providing ``growth_rate`` and ``dz_norm_lcdm_interp``.
          nuisance_parameters (dict): Bias and counterterm parameters.
          redshift (float): Single redshift at which to evaluate PBJ.
        """
        self.linear_perturbations_lcdm = linear_perturbations_lcdm
        self.linear_perturbations_darkenergy = linear_perturbations_darkenergy
        self.background = linear_perturbations_darkenergy.background
        self.parameters = nuisance_parameters

        assert np.asarray(redshift).size == 1, "Only a single redshift can be passed."
        assert redshift in linear_perturbations_lcdm.z, (
            "Redshift requested for PBJ not previously computed with linear theory code"
        )

        self.redshift = redshift

        self.cosmo = {
            "h": self.background.h,
            "Och2": self.background.Omega_cdm0 * self.background.h**2,
            "Obh2": self.background.Omega_b0 * self.background.h**2,
            "As": self.background.As,
            "ns": self.background.ns,
            "Mnu": self.background.mnu,
            "w0": -1.0,
            "wa": 0.0,
            "Tcmb": 2.7255,
        }

    def _de_growth(self) -> tuple[float, float]:
        """Dark-energy growth rate and LCDM-normalized growth factor.

        Binned-w growth is scale independent. ``0.05`` 1/Mpc only selects a
        point on that spline. A scalar redshift returns shape ``(1, 1)``.
        """
        de = self.linear_perturbations_darkenergy
        f = np.squeeze(de.growth_rate(self.redshift, 0.05))
        # D_de(z) / D_LCDM(0), matching the LCDM P_L(k, z=0) used in the kernels.
        D = np.squeeze(de.dz_norm_lcdm_interp(self.redshift, 0.05))
        return f, D

    def _linear_spectrum(self, kind: str) -> np.ndarray:
        """LCDM P(k, z=0) on the PBJ wavenumber grid, squeezed to 1d."""
        spectrum = (
            self.linear_perturbations_lcdm.matter_power_spectrum_cb
            if kind == "cb"
            else self.linear_perturbations_lcdm.matter_power_spectrum
        )
        return np.squeeze(
            spectrum(0.0, pbj_obj.kL, hubble_units=False, k_hunit=False)
        )

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight

        Returns:
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        plinear = self._linear_spectrum("cb")
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")
        f, D = self._de_growth()

        pkmu = pbj_obj.P_kmu_2D(
            self.redshift,
            True,
            kgrid=k,
            mu=mu,
            f=f,
            D=D,
            cosmo=self.cosmo,
            IRres=True,
            **self.parameters,
        )

        return pkmu

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
            2D power spectrum of specific terms
        """
        plinear = self._linear_spectrum("total")
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")
        f, D = self._de_growth()

        pkmu_marg_dict = pbj_obj.P_kmu_2D_marg_dict(
            self.redshift,
            True,
            kgrid=k,
            mu=mu,
            f=f,
            D=D,
            cosmo=self.cosmo,
            IRres=True,
            b1=self.parameters["b1"],
        )
        Pk2d = np.array([pkmu_marg_dict[key] for key in term_list])

        return Pk2d
