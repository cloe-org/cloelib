"""Interface of Legendre Multipoles with PBJ."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
import numpy as np  # type: ignore

try:
    from pbjcosmo.theory import Theory

    pbj_obj = Theory()
except (ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"PBJ could not be imported or initialised: {e}")


class PBJSpectroPower:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PBJ."""

    NLcode = "PBJ"

    def __init__(
        self,
        linear_perturbations: Perturbations,
        nuisance_parameters: dict,
        redshift: float,
    ):
        r"""Class constructor.

        Args:
          linear_perturbations (Perturbations): Perturbations object containing cosmology, linear power spectrum,
            and growth functions
          nuisance_parameters (dict): Dictionary containing bias and counterterm parameters
          redshift (float): single redshift in which to evaluate PBJ
        """
        self.linear_perturbations = linear_perturbations
        self.background = linear_perturbations.background
        self.parameters = nuisance_parameters

        assert np.asarray(redshift).size == 1, "Only a single redshift can be passed."
        assert redshift in linear_perturbations.z, (
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
            "w0": self.background.w0,
            "wa": self.background.wa,
            "Tcmb": 2.7255,
        }

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight

        Returns:
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        plinear = self.linear_perturbations.matter_power_spectrum_cb(
            0.0, pbj_obj.kL, hubble_units=False, k_hunit=False
        )
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")

        pkmu = pbj_obj.P_kmu_2D(
            self.redshift,
            True,
            kgrid=k,
            mu=mu,
            f=self.linear_perturbations.growth_rate()[
                self.linear_perturbations.z == self.redshift
            ],
            D=self.linear_perturbations.growth_factor_cb(self.redshift, 0.05),
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
        plinear = self.linear_perturbations.matter_power_spectrum(
            0.0, pbj_obj.kL, hubble_units=False, k_hunit=False
        )
        pbj_obj._Pgg_kmu_terms(plinear, self.cosmo, units="1/Mpc")

        f = float(
            np.squeeze(
                self.linear_perturbations.growth_rate()[
                    self.linear_perturbations.z == self.redshift
                ]
            )
        )
        D = float(
            np.squeeze(
                self.linear_perturbations.growth_factor(self.redshift, 0.05)
            )
        )
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