"""Interface of Legendre Multiples with Comet."""

# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
import numpy as np  # type: ignore

# Cosmology imports
try:
    from comet import comet  # type: ignore

    comet_inst = comet(model="EFT", use_Mpc=True, bias_basis="AssBauGre")
except ImportError:
    raise ImportError("Comet could not be imported or initialised.")


class CometEFT_SpectroPower:
    r"""Class to retrieve :math:`P(k,\mu)` (including RSD) with the EFT model from COMET."""

    def __init__(self, background: Background, RSD_parameters: dict, redshift: float):
        r"""Class constructor.

        Parameters
        ----------
        background: Background
            Background class containing cosmology and background distances
        RSD_parameters: dict
            Dictionary containing bias and counterterm parameters
        redshift: float
            Redshift at which to evaluate :math:`P(k,\mu)`
        """
        self.background = background

        self.parameters = {}
        self.parameters["wc"] = self.background.Omega_cdm0 * self.background.h**2
        self.parameters["wb"] = self.background.Omega_b0 * self.background.h**2
        self.parameters["Mnu"] = self.background.mnu
        self.parameters["ns"] = self.background.ns
        self.parameters["h"] = self.background.h
        self.parameters["As"] = self.background.As * 1e9
        self.parameters["w0"] = self.background.w0
        self.parameters["wa"] = self.background.wa
        self.parameters["Ok"] = self.background.Omega_k0
        self.parameters.update(RSD_parameters)
        self.parameters["z"] = redshift

        self.redshift = redshift

        self.diagram_naming_relation = {
            "b1-b1": ["P0L_b1b1", "P1L_b1b1"],
            "b1-b2": "P1L_b1b2",
            "b1-bG2": "P1L_b1g2",
            "b1-bGam3": "P1L_b1g21",
            "b2-b2": "P1L_b2b2",
            "b2-bG2": "P1L_b2g2",
            "bG2-bG2": "P1L_g2g2",
            "b1": "PNL_b1",
            "b2": "P1L_b2",
            "bG2": "P1L_g2",
            "bGam3": "P1L_g21",
            "v-v": "PNL_id",
            "c0": "Pctr_c0",
            "c2": "Pctr_c2",
            "c4": "Pctr_c4",
            "b1-b1-cnlo": "Pctr_b1b1cnlo",
            "b1-cnlo": "Pctr_b1cnlo",
            "cnlo": "Pctr_cnlo",
        }

    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""2D power spectrum from couplings of density and velocity fields.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        Pk2d_rsd: np.ndarray
            2D power spectrum from couplings of density and velocity fields
        """
        return np.squeeze(
            comet_inst.P2d_nostoch(
                k=k[:, :, np.newaxis],
                mu=mu[:, np.newaxis],
                params=self.parameters,
                de_model="w0wa",
            )
        )

    def Pk2d_term_rsd(
        self, k: np.ndarray, mu: np.ndarray, term_list: list
    ) -> np.ndarray:
        r"""2D power spectrum for a subset of specific diagrams of the loop expansion.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        term_list: list
            Identifiers of loop diagrams
        Returns
        -------
        Pk2d_term_rsd: np.ndarray
            2D power spectrum of specific terms
        """
        term_list_expanded, index_map = [], {}
        for key in term_list:
            vals = self.diagram_naming_relation[key]
            vals = vals if isinstance(vals, list) else [vals]
            index_map[key] = list(
                range(len(term_list_expanded), len(term_list_expanded) + len(vals))
            )
            term_list_expanded.extend(vals)
        Pk2d_expanded = np.squeeze(
            comet_inst.PX_2d(
                k=k[:, :, np.newaxis],
                mu=mu[:, np.newaxis],
                params=self.parameters,
                X_list=term_list_expanded,
                de_model="w0wa",
            ),
            axis=-1,
        )
        Pk2d = np.array(
            [
                (
                    np.sum(Pk2d_expanded[indices], axis=0)
                    if len(indices) > 1
                    else Pk2d_expanded[indices[0]]
                )
                for key, indices in index_map.items()
            ]
        )
        return Pk2d
