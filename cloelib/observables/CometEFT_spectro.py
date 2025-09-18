"""Interface of Legendre Multiples with Comet."""

# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Protocol, Union, TypeVar, Optional, Sequence
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
        self.parameters["Mnu"] = self._set_neutrino_parameters(self.background)
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

    def _set_neutrino_parameters(self, background: Background) -> float:
        r"""Set neutrino parameters in the parameters dictionary.

        This method adds neutrino parameters to the provided dictionary.
        It also ensures consistency with the background cosmology.
        Comet only supports a single species of neutrinos, so this method
        throws an error if multiple neutrino species are provided.
        Note that Comet supports mnu=0.0 (N_mnu=0).

        Parameters
        ----------
        parameters: dict
            Dictionary to which neutrino parameters will be added
        """
        if background.N_mnu > 1:
            raise ValueError("Comet supports max a single species of neutrinos. "
                             "Set N_mnu=1 in the Background class.")
        if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
            raise ValueError(
                "Comet only supports a fixed number of effective" 
                f"relativistic species (N_eff=3.044). Found {background.N_eff} "
                "Ensure that N_eff=3.044 in the Background class."
            )
        if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
            raise ValueError(
                "Comet only supports a single species of neutrinos. "
                "Set N_mnu=1 in the Background class."
            )
        else:
            mnu_arg = float(background.mnu)
        # returns the neutrino mass in eV
        return mnu_arg

    def _set_neutrino_parameters(self, background: Background) -> float:
        r"""Set neutrino parameters in the parameters dictionary.

        This method adds neutrino parameters to the provided dictionary.
        It also ensures consistency with the background cosmology.
        Comet only supports a single species of neutrinos, so this method
        throws an error if multiple neutrino species are provided.
        Note that Comet supports mnu=0.0 (N_mnu=0).

        Parameters
        ----------
        parameters: dict
            Dictionary to which neutrino parameters will be added
        """
        if background.N_mnu > 1:
            raise ValueError("Comet supports max a single species of neutrinos. "
                             "Set N_mnu=1 in the Background class.")
        if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
            raise ValueError(
                "Comet only supports a fixed number of effective" 
                f"relativistic species (N_eff=3.044). Found {background.N_eff} "
                "Ensure that N_eff=3.044 in the Background class."
            )
        if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
            raise ValueError(
                "Comet only supports a single species of neutrinos. "
                "Set N_mnu=1 in the Background class."
            )
        else:
            mnu_arg = float(background.mnu)
        # returns the neutrino mass in eV
        return mnu_arg

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
