"""Module to compute Legendre multipoles in presence of interlopers."""

# cloelib imports
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles

# General imports
from typing import Optional
import numpy as np

class ContaminatedLegendreMultipoles:
    """Class to compute spectroscopic Legendre multipoles in presence of interlopers."""

    def __init__(
        self,
        species: dict[str, tuple[LegendreMultipoles, float]],
    ):
        """Initialize the class instance.

        Parameters
        ----------
        species : dict
            Dictionary containing the different species of contaminants, each
            one of them being a tuple with the corresponding LegendreMultipoles
            object and the fraction
        """
        self.species = species
        self.correct_redshift = species['correct'][0].redshift
        self.ap_distortion = species['correct'][0].ap_distortion

    def power_multipoles(
        self, k: np.ndarray,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles in presence of interlopers.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        gamma_dict = {
            key: [self.ap_distortion.gamma_tr(val[0].redshift, self.correct_redshift),
                  self.ap_distortion.gamma_lo(val[0].redshift, self.correct_redshift)]
            for key, val in self.species.items()}
        multipoles_spec = {
            key: val[0].power_multipoles(
                k=k, ells=ells, use_AP=use_AP, gamma=gamma_dict[key])
            for key, val in self.species.items()
        }
        return {f'ell{ell}': np.sum(np.stack([
                    multipoles_spec[key][f"ell{ell}"] * val[1]**2
                    for key, val in self.species.items()]), axis=0)
                for ell in ells}

    def convolved_power_multipoles(
        self, mixing_matrix: dict,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True
    ) -> dict:
        r"""Power spectrum Legendre multipoles convolved with the mixing matrix
            in presence of interlopers.

        Parameters
        ----------
        mixing_matrix: dict
            Dictionary containing the mixing matrices of the different species
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles_out: dict
            Dictionary containing the output wavemode array (read from the
            mixing matrix dictionary) and the convolved Legendre multipoles
        """
        gamma_dict = {
            key: [self.ap_distortion.gamma_tr(val[0].redshift, self.correct_redshift),
                  self.ap_distortion.gamma_lo(val[0].redshift, self.correct_redshift)]
            for key, val in self.species.items()}
        multipoles_spec = {
            key: val[0].convolved_power_multipoles(
                mixing_matrix=mixing_matrix[key], ells=ells,
                use_AP=use_AP, gamma=gamma_dict[key])
            for key, val in self.species.items()}
        return {f'ell{ell}': np.sum(np.stack([
            multipoles_spec[key][f"ell{ell}"] * val[1]**2
            for key, val in self.species.items()]), axis=0)
        for ell in ells}

    def power_term_multipoles(
        self, k: np.ndarray, term_list: list,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True) -> dict:
        r"""Power spectrum Legendre multipoles of specified terms in presence
            of interlopers.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        term_list: list
            List of terms to compute
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        gamma_dict = {
            key: [self.ap_distortion.gamma_tr(val[0].redshift, self.correct_redshift),
                  self.ap_distortion.gamma_lo(val[0].redshift, self.correct_redshift)]
            for key, val in self.species.items()}
        multipoles_spec = {
            key: val[0].power_term_multipoles(
                k=k, term_list=term_list, ells=ells,
                use_AP=use_AP, gamma=gamma_dict[key])
            for key, val in self.species.items()
        }
        return {f'ell{ell}': np.stack([
                    multipoles_spec[key][f"ell{ell}"] * val[1]**2
                    for key, val in self.species.items()]) for ell in ells}

    def convolved_power_term_multipoles(
        self, mixing_matrix: dict, term_list: list,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True
    ) -> dict:
        r"""Power spectrum Legendre multipoles of the specified terms convolved
            with the mixing matrix in presence of interlopers.

        Parameters
        ----------
        mixing_matrix: dict
            Dictionary containing the mixing matrices of the different species
        term_list: list
            List of terms to compute
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        Returns
        -------
        multipoles_out: dict
            Dictionary containing the output wavemode array (read from the
            mixing matrix dictionary) and the convolved Legendre multipoles
        """
        gamma_dict = {
            key: [self.ap_distortion.gamma_tr(val[0].redshift, self.correct_redshift),
                  self.ap_distortion.gamma_lo(val[0].redshift, self.correct_redshift)]
            for key, val in self.species.items()}
        multipoles_spec = {
            key: val[0].convolved_power_term_multipoles(
                mixing_matrix=mixing_matrix[key], term_list=term_list,
                ells=ells, use_AP=use_AP, gamma=gamma_dict[key])
            for key, val in self.species.items()}
        return {f'ell{ell}': np.stack([
            multipoles_spec[key][f"ell{ell}"] * val[1]**2
            for key, val in self.species.items()]) for ell in ells}
