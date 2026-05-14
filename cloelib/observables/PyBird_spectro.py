"""Interface of Legendre Multipoles with PyBIrd."""

# cloelib imports
from cloelib.cosmology.cosmology import Perturbations

# General imports
import numpy as np  # type: ignore
from scipy.interpolate import make_interp_spline
from scipy.special import legendre

try:
    from pybird.correlator import Correlator

    N = Correlator()
    N.set(
        {
            "output": "bPk",
            "multipole": 3,
            "kmax": 0.5,
            "km": 1.0,
            "kr": 1.0,
            "eft_basis": "pbj",
            "with_bias": False,
            "with_stoch": False,
            "with_resum": True,
            "optiresum": True,
            "with_nnlo_counterterm": True,
        }
    )
    ells = [0, 2, 4]

except (ImportError, AttributeError, TypeError) as e:
    raise ImportError(f"PyBird could not be imported or initialised: {e}")


class PyBirdSpectroPower:
    r"""Class to retrieve $P(k,\mu)$ with the EFT model from PyBird"""

    def __init__(self, linear_perturbations: Perturbations, nuisance_parameters: dict):
        r"""Class constructor.

        Args:
          linear_perturbations (Perturbations): Perturbations object containing cosmology, linear power spectrum,
            redshift and growth functions
          nuisance_parameters (dict): Dictionary containing bias and counterterm parameters
        """
        self.linear_perturbations = linear_perturbations
        self.background = linear_perturbations.background
        self.parameters = nuisance_parameters
        self.mask_z0 = linear_perturbations.z != 0.0
        self.redshift = linear_perturbations.z[self.mask_z0]
        z = self.redshift[0]  # assuming one sky - one redshift for now

        kk = np.geomspace(1.0e-4, 2.0, 512)
        plin = self.linear_perturbations.matter_power_spectrum(
            z, kk, hubble_units=True, k_hunit=True
        )
        h = self.background.h
        f = self.linear_perturbations.growth_rate()[self.mask_z0][0]
        D = self.linear_perturbations.growth_factor(z, 0.05)  # kpivot = 0.05

        N.compute({"kk": kk, "pk_lin": plin, "z": z, "D": D, "f": f})
        k_ = N.co.k  # [h/Mpc]

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

        marg_parameter_names = {"bGamma3", "c0", "c2", "c4", "ct"}

        pkl_term = N.getmarg(
            nuisance_parameters_pybird,
            marg_parameter_names,
        ).reshape(-1, 3, k_.shape[0])  # shape (Nterm, Nk*Nl) -> (Nterm, Nl, Nk)

        for i, c in enumerate(marg_parameter_names):
            if c in ["c0", "c2", "c4"]:
                pkl_term[i] *= h**2  # [Mpc]^2 -> [Mpc/h]^2
            if c in ["ct"]:
                pkl_term[i] *= (
                    -(h**4)
                )  # [Mpc]^4 -> [Mpc/h]^4 + sign sitch to match PBJ convention

        self.ipkl_term = make_interp_spline(
            k_ * h, pkl_term / h**3, k=3, axis=-1
        )  # k: [1/Mpc], Pk: [Mpc]^3

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

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Args:
          k (np.ndarray): Wavenumber
          mu (np.ndarray): Angle (cosinus) to the line of sight

        Returns:
          Pk2d_rsd (np.ndarray): 2D power spectrum from couplings of density and velocity fields
        """
        leglmu = np.array([legendre(ell)(mu) for ell in ells])
        if k.ndim == 1:
            sumrule = "lk,lm->km"
        elif k.ndim == 2 and k.shape[1] == mu.shape[0]:
            sumrule = "lkm,lm->km"  # for AP effect, k is a 2D mesh
        pkmu = np.einsum(sumrule, self.ipkl(k), leglmu)
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
        leglmu = np.array([legendre(ell)(mu) for ell in ells])
        pkmu_term = np.einsum("nlk,lm->nkm", self.ipkl_term(k), leglmu)
        return pkmu_term
