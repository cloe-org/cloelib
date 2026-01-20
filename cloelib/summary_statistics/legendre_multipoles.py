"""Module to compute Legendre multipoles."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.observables.spectro import SpectroPower
from cloelib.summary_statistics.APDistortion import APDistortion
from cloelib.auxiliary.math_utils import legendre
from cloelib.auxiliary.fftlog import fftlog
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
from typing import Optional
import numpy as np


class LegendreMultipoles:
    """Class to compute spectroscopic Legendre multipoles of the galaxy power spectrum."""

    def __init__(
        self,
        spectro_power: SpectroPower,
        background_fiducial: Background,
        nbar: float,
    ):
        """Initialize the class instance.

        Parameters
        ----------
        spectro_power: SpectroPower
            Class returning the anisotropic power spectrum (only density and
            velocity field couplings; noise and systematics are included directly
            here)
        background_fiducial: Background
            Background class for computing fiducial background distances
        nbar: float
            Mean number denisty of the sample
        """
        self.spectro_power = spectro_power
        self.background_fiducial = background_fiducial
        self.ap_distortion = APDistortion(spectro_power.background, background_fiducial)

        self.mu_grid, self.mu_weights = np.polynomial.legendre.leggauss(10)
        self.mu_grid = 0.5 * (self.mu_grid + 1.0)
        self.mu_weights *= 0.5

        self.nbar = nbar

    def _ensure_array(self, param):
        """Ensure that the input parameter is a NumPy array.

        If the input is a scalar, it is converted to a NumPy array.

        Parameters
        ----------
        param : scalar or array-like
            Input parameter.

        Returns
        -------
        numpy.ndarray
            Input parameter as a NumPy array.
        """
        if np.isscalar(param):
            param = np.array([param])
        return np.asarray(param)

    def _k_AP(
        self, k: np.ndarray, mu: np.ndarray, z: float,
        use_AP: Optional[bool] = True, gamma: Optional[list] = None
    ) -> np.ndarray:
        r"""AP-distorted wavenumber.

        .. math::
            k(k_{\rm fid},\mu_{\rm fid}, z) &= k_{\rm fid} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{1/2}
        Parameters
        ----------
        k: np.ndarray
           Fiducial wavenumber
        mu: np.ndarray
           Fiducial angle (cosinus) to the line of sight
        z: float
           Redshift
        use_AP: bool
            Flag to switch between with and without AP corrections
        gamma: list, optional
            Values of rescaling factors due to line misidentification for the
            direction perpendicular and parallel to the line of sight (
            default is None)
        Returns
        -------
        kAP: np.ndarray
           AP-distorted wavenumber
        """
        q_tr = self.ap_distortion.q_AP_tr(z) if use_AP else 1.0
        q_lo = self.ap_distortion.q_AP_lo(z) if use_AP else 1.0
        if gamma: q_tr, q_lo = (q_tr * gamma[0], q_lo * gamma[1])
        return np.outer(k, np.sqrt(mu**2 / q_lo**2 + (1.0 - mu**2) / q_tr**2))

    def _mu_AP(
        self, mu: np.ndarray, zs: float, use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ) -> np.ndarray:
        r"""AP-distorted angle (cosinus) to the line of sight.

        .. math::
            \mu(\mu_{\rm fid}, z) &= \frac{\mu_{\rm fid}}{q_\parallel(z)} \
            \left[\frac{(\mu_{\rm fid})^2}{q_\parallel^2(z)} + \
            \frac{1-(\mu_{\rm fid}^2)}{q_\perp^2(z)}\right]^{-1/2}
        Parameters
        ----------
        mu: np.ndarray
           Fiducial angle (cosinus) to the line of sight
        z: float
           Redshift
        use_AP: bool
            Flag to switch between with and without AP corrections
        gamma: list, optional
            Values of rescaling factors due to line misidentification for the
            direction perpendicular and parallel to the line of sight (
            default is None)
        Returns
        -------
        muAP: np.ndarray
           AP-distorted angle (cosinus) to the line of sight
        """
        q_tr = self.ap_distortion.q_AP_tr(zs) if use_AP else 1.0
        q_lo = self.ap_distortion.q_AP_lo(zs) if use_AP else 1.0
        if gamma: q_tr, q_lo = (q_tr * gamma[0], q_lo * gamma[1])
        return mu / q_lo / np.sqrt(mu**2 / q_lo**2 + (1.0 - mu**2) / q_tr**2)

    def _damping_function(
        self, k: np.ndarray, mu: np.ndarray, z: float,
        noise_syst_parameters: dict
    ) -> np.ndarray:
        r"""Damping function due to GCsp redshift uncertainty.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        z: float
            Redshift
        noise_syst_parameters: dict
            Dictionary containing shot noise and parameters related to
            observational systematics
        Returns
        -------
        damping_function: np.ndarray
            Damping function due to GCsp redshift uncertainty
        """
        sigma_z = noise_syst_parameters["sigmaz"]
        sigma_r = (SPEED_OF_LIGHT / 1000.0 * sigma_z /
                   self.background_fiducial.hubble_parameter(z))
        return np.exp(-k**2 * mu**2 * sigma_r**2)

    def _Pk2d_noise(self, k: np.ndarray, mu: np.ndarray,
                    noise_syst_parameters: dict) -> np.ndarray:
        r"""2D power spectrum from expansion of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        noise_syst_parameters: dict
            Dictionary containing shot noise and parameters related to
            observational systematics
        Returns
        -------
        Pk2d_noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = (
            noise_syst_parameters["NP0"] * self._Pk2d_noise_k0(k)
            + noise_syst_parameters["NP20"] * self._Pk2d_noise_k0(k)
            + noise_syst_parameters["NP22"] * self._Pk2d_noise_k2mu2(k, mu)
        )
        return noise

    def _Pk2d_noise_k0(self, k: np.ndarray) -> np.ndarray:
        r"""Leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = np.full_like(k, 1.0)
        return noise / self.nbar

    def _Pk2d_noise_k2(self, k: np.ndarray) -> np.ndarray:
        r"""Isotropic next-to-leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2
        return noise / self.nbar

    def _Pk2d_noise_k2mu2(self, k: np.ndarray, mu: np.ndarray) -> np.ndarray:
        r"""Anisotropic next-to-leading-order term from 2d power spectrum of stochastic field.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        Returns
        -------
        noise: np.ndarray
            2D power spectrum from expansion of stochastic field
        """
        noise = k**2 * legendre(2, mu)
        return noise / self.nbar

    def _Pk2d_tot(
        self, k: np.ndarray, mu: np.ndarray, z: float,
        RSD_parameters: dict, noise_syst_parameters: dict
    ) -> np.ndarray:
        r"""Total 2D power spectrum (including RSD, systematics, and noise).

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        mu: np.ndarray
            Angle (cosinus) to the line of sight
        z: float
           Redshift
        RSD_parameters: dict
            Dictionary containing bias and counterterm parameters
        noise_syst_parameters: dict
            Dictionary containing shot noise and parameters related to
            observational systematics
        Returns
        -------
        Pk2d_tot: np.ndarray
            Total 2D power spectrum (including RSD, systematics, and noise)
        """
        return (self.spectro_power.Pk2d_rsd(k, mu, z, RSD_parameters) *
                self._damping_function(k, mu, z, noise_syst_parameters) *
                (1.0 - noise_syst_parameters["fout"]) ** 2 +
                self._Pk2d_noise(k, mu, noise_syst_parameters))

    def power_multipoles(
        self, k: np.ndarray, z: float,
        RSD_parameters: Optional[dict] = None,
        noise_syst_parameters: Optional[dict] = None,
        species: Optional[dict] = None,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ) -> dict:
        r"""Power spectrum Legendre multipoles.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        z: float
            Observed redshift
        RSD_parameters: dict, optional
            Dictionary of bias and counterterm parameters (default is None)
        noise_syst_parameters: dict, optional
            Dictionary of shot noise and systematics parameters (default is
            None)
        species: dict, optional
            Dictionary containing the different samples (main and contaminants)
            with each element being a subdictionary containing redshift,
            fraction, and nuisance parameters (default is None)
        ells: np.ndarray, optional
            Legendre multipole order (default is [0, 2, 4])
        use_AP: bool, optional
            Flag to switch between with and without AP corrections (default is
            True)
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])

        if species is not None:
            gamma_dict = {
                key: [self.ap_distortion.gamma_tr(val["z"], z),
                      self.ap_distortion.gamma_lo(val["z"], z)]
                for key, val in species.items()}

            multipoles_spec = {key: self.power_multipoles(
                k=k, z=val["z"], RSD_parameters=val["RSD_parameters"],
                noise_syst_parameters=val["noise_syst_parameters"],
                ells=ells, use_AP=use_AP, gamma=gamma_dict[key])
                for key, val in species.items()}

            return {f'ell{ell}': np.sum(np.stack([
                        multipoles_spec[key][f"ell{ell}"] * val["fraction"]**2
                        for key, val in species.items()]), axis=0)
                    for ell in ells}

        AP_factor = (
            self.ap_distortion.q_AP_tr(z) ** 2
            * self.ap_distortion.q_AP_lo(z)
            if use_AP
            else 1.0
        )
        if gamma: AP_factor *= gamma[0]**2 * gamma[1]
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / AP_factor
        kAP = self._k_AP(k, self.mu_grid, z, use_AP=use_AP, gamma=gamma)
        muAP = self._mu_AP(self.mu_grid, z, use_AP=use_AP, gamma=gamma)
        Pk2d_tot = self._Pk2d_tot(kAP, muAP, z, RSD_parameters,
                                  noise_syst_parameters)
        multipoles = {}
        for i, ell in enumerate(ells):
            leg = legendre(ell, self.mu_grid)
            multipoles[f"ell{ell}"] = np.einsum(
                "ab,b,b->a", Pk2d_tot, leg, self.mu_weights
            )
            multipoles[f"ell{ell}"] *= 2.0 * prefactors[i]
        return multipoles

    def _input_power_multipoles(
        self, kin_arrays: dict, z: float,
        RSD_parameters: Optional[dict] = None,
        noise_syst_parameters: Optional[dict] = None,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ):
        r"""Input multipoles for convolution with mixing matrix

        Parameters
        ----------
        kin_arrays: dict
            Dictionary containing the input wavemodes for each Legendre
            multipole (0, 2, 4) (comes from the mixing matrix dictionary
            read in `convolved_power_multipoles`)
        z: float
            Redshift
        RSD_parameters: dict, optional
            Dictionary containing bias and counterterm parameters (default is
            None)
        noise_syst_parameters: dict, optional
            Dictionary containing shot noise and parameters related to
            observational systematics (default is None)
        use_AP : bool, optional
            Flag to switch between with and without AP corrections (default is
            True)
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)

        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles
        """
        ells_tot = [0, 2, 4]

        if all(np.array_equal(kin_arrays[0], kin) for kin in kin_arrays):
            return self.power_multipoles(
                    k=kin_arrays[0], z=z, RSD_parameters=RSD_parameters,
                    noise_syst_parameters=noise_syst_parameters,
                    ells=ells_tot, use_AP=use_AP, gamma=gamma)
        else:
            return {
                f"ell{ell}": (
                    self.power_multipoles(
                        k=kin_arrays[i], z=z, RSD_parameters=RSD_parameters,
                        noise_syst_parameters=noise_syst_parameters,
                        ells=[ell], use_AP=use_AP, gamma=gamma)[f"ell{ell}"])
                for i, ell in enumerate(ells_tot)}

    def _convolve_with_mixing_matrix(self, mixing_matrix: dict,
                                     multipoles_in: dict,
                                     ells: np.ndarray):
        r"""Convolution of input Legendre multipoles with mixing matrix

        Parameters
        ----------
        mixing_matrix: dict
            Dictionary containing the mixing matrix (comes from the mixing
            matrix dictionary read in `convolved_power_multipoles`)
        multipoles_in: dict
            Dictionary containing the input Legendre multipoles
        ells: np.ndarray
            Order of convolved Legendre multipoles

        Returns
        -------
        multipoles_out: dict
            Dictionary containing the convolved Legendre multipoles
        """
        return {f"ell{ell}": sum(np.dot(mixing_matrix[f"W{ell}{ell_prime}"],
                                        multipoles_in[f"ell{ell_prime}"])
                                 for ell_prime in [0, 2, 4]) for ell in ells}

    def convolved_power_multipoles(
        self, z: float, mixing_matrix: Optional[dict] = None,
        RSD_parameters: Optional[dict] = None,
        noise_syst_parameters: Optional[dict] = None,
        species: Optional[dict] = None,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ) -> dict:
        r"""Power spectrum Legendre multipoles convolved with the mixing matrix.

        Parameters
        ----------
        z: float
            Redshift
        mixing_matrix: dict
            Dictionary containing the mixing matrix
            (used if `species` dictionary is None)
        RSD_parameters: dict, optional
            Dictionary of bias and counterterm parameters
            (used if `species` keyword is None)
        noise_syst_parameters: dict, optional
            Dictionary of shot noise and systematics parameters
            (used if `species` keyword is None)
        species: dict, optional
            Dictionary of different samples (main and contaminants), including
            the mixing matrix, effective redshift, fraction, and nuisance
            parameters for each
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)
        Returns
        -------
        multipoles_out: dict
            Dictionary containing the output wavemode array (read from the
            mixing matrix dictionary) and the convolved Legendre multipoles
        """
        ells_tot = [0, 2, 4]
        ells = self._ensure_array(ells) if ells is not None else ells_tot

        if species is not None:
            gamma_dict = {
                key: [self.ap_distortion.gamma_tr(val["z"], z),
                      self.ap_distortion.gamma_lo(val["z"], z)]
                for key, val in species.items()}

            multipoles_spec = {}

            for key, val in species.items():
                if "mixing_matrix" not in val:
                    raise ValueError(
                        f"Missing mixing matrix for species '{key}'"
                )

                multipoles_spec[key] = self.convolved_power_multipoles(
                    z=val["z"], mixing_matrix=val["mixing_matrix"],
                    RSD_parameters=val["RSD_parameters"],
                    noise_syst_parameters=val["noise_syst_parameters"],
                    species=None, ells=ells, use_AP=use_AP, gamma=gamma_dict[key]
                )

            return {
                f"ell{ell}": np.sum(
                    np.stack([
                        multipoles_spec[key][f"ell{ell}"] * val["fraction"]**2
                        for key, val in species.items()
                    ]), axis=0,
                )
                for ell in ells
            }

        if mixing_matrix is None:
            raise ValueError(
                "Mixing matrix must be provided when species is None"
            )

        kin_arrays = [mixing_matrix[f"kin{ell}"] for ell in ells_tot]

        multipoles_in = self._input_power_multipoles(
            kin_arrays=kin_arrays, z=z,
            RSD_parameters=RSD_parameters,
            noise_syst_parameters=noise_syst_parameters,
            use_AP=use_AP, gamma=gamma)

        multipoles_out = self._convolve_with_mixing_matrix(
            mixing_matrix, multipoles_in, ells)

        return {"k": mixing_matrix["kout"], **multipoles_out}

    def power_term_multipoles(
        self, k: np.ndarray, z: float, term_list: list,
        species: Optional[dict] = None,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ) -> dict:
        r"""Power spectrum Legendre multipoles of specified terms.

        Parameters
        ----------
        k: np.ndarray
            Wavenumber
        z: float
            Redshift
        term_list: list
            List of terms to compute
        species: dict, optional
            Dictionary containing the different samples (main and contaminants)
            with each element being a subdictionary containing redshift,
            fraction, and nuisance parameters (default is None)
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)
        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles of specified terms
        """
        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])

        if species is not None:
            gamma_dict = {
                key: [self.ap_distortion.gamma_tr(val["z"], z),
                      self.ap_distortion.gamma_lo(val["z"], z)]
                for key, val in species.items()}

            multipoles_spec = {key: self.power_term_multipoles(
                k=k, z=val["z"], term_list=term_list,
                ells=ells, use_AP=use_AP, gamma=gamma_dict[key])
                for key, val in species.items()}

            return {f'ell{ell}': np.stack([
                        multipoles_spec[key][f"ell{ell}"] * val["fraction"]**2
                        for key, val in species.items()]) for ell in ells}

        AP_factor = (
            self.ap_distortion.q_AP_tr(z) ** 2
            * self.ap_distortion.q_AP_lo(z)
            if use_AP
            else 1.0
        )
        if gamma: AP_factor *= gamma[0]**2 * gamma[1]
        prefactors = np.array([(2.0 * m + 1.0) for m in ells]) / 2.0 / AP_factor
        kAP = self._k_AP(k, self.mu_grid, z, use_AP=use_AP, gamma=gamma)
        muAP = self._mu_AP(self.mu_grid, z, use_AP=use_AP, gamma=gamma)
        Pk2d = np.empty((len(term_list), len(k), len(self.mu_grid)))
        rsd_ids = [index for index, term in enumerate(term_list) if "noise" not in term]
        if rsd_ids:
            Pk2d[rsd_ids] = self.spectro_power.Pk2d_term_rsd(
                kAP, muAP, z=z, term_list=[term_list[index] for index in rsd_ids]
            )
        noise_ids = [index for index in range(len(term_list)) if index not in rsd_ids]
        if noise_ids:
            noise_func = {
                "noise_k0": self._Pk2d_noise_k0,
                "noise_k2": self._Pk2d_noise_k2,
                "noise_k2mu2": self._Pk2d_noise_k2mu2,
            }
            Pk2d[noise_ids] = np.array(
                [
                    (
                        noise_func[term_list[index]](kAP, muAP)
                        if term_list[index] == "noise_k2mu2"
                        else noise_func[term_list[index]](kAP)
                    )
                    for index in noise_ids
                ]
            )
        multipoles = {}
        for i, ell in enumerate(ells):
            leg = legendre(ell, self.mu_grid)
            multipoles[f"ell{ell}"] = np.einsum(
                "abc,c,c->ab", Pk2d, leg, self.mu_weights
            )
            multipoles[f"ell{ell}"] *= 2.0 * prefactors[i]
        return multipoles

    def _input_power_term_multipoles(
        self, kin_arrays: dict, z: float, term_list: list,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ):
        r"""Input multipoles for convolution with mixing matrix

        Parameters
        ----------
        kin_arrays: dict
            Dictionary containing the input wavemodes for each Legendre
            multipole (0, 2, 4) (comes from the mixing matrix dictionary
            read in `convolved_power_multipoles`)
        z: float
            Redshift
        term_list: list
            List of terms to compute
        use_AP : bool, optional
            Flag to switch between with and without AP corrections (default is
            True)
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)

        Returns
        -------
        multipoles: dict
            Power spectrum Legendre multipoles for specific terms
        """
        ells_tot = [0, 2, 4]

        if all(np.array_equal(kin_arrays[0], kin) for kin in kin_arrays):
            return self.power_term_multipoles(
                    k=kin_arrays[0], z=z, term_list=term_list,
                    ells=ells_tot, use_AP=use_AP, gamma=gamma)
        else:
            return {
                f"ell{ell}": (
                    self.power_term_multipoles(
                        k=kin_arrays[i], z=z, term_list=term_list,
                        ells=[ell], use_AP=use_AP, gamma=gamma)[f"ell{ell}"])
                for i, ell in enumerate(ells_tot)}

    def _convolve_terms_with_mixing_matrix(self, mixing_matrix: dict,
                                           multipoles_in: dict,
                                           ells: np.ndarray):
        r"""Convolution of input Legendre multipoles with mixing matrix

        Parameters
        ----------
        mixing_matrix: dict
            Dictionary containing the mixing matrix (comes from the mixing
            matrix dictionary read in `convolved_power_multipoles`)
        multipoles_in: dict
            Dictionary containing the input Legendre multipoles
        ells: np.ndarray
            Order of convolved Legendre multipoles

        Returns
        -------
        multipoles_out: dict
            Dictionary containing the convolved Legendre multipoles
        """
        return {f"ell{ell}": sum(np.dot(mixing_matrix[f"W{ell}{ell_prime}"],
                                        multipoles_in[f"ell{ell_prime}"].T).T
                                 for ell_prime in [0, 2, 4]) for ell in ells}

    def convolved_power_term_multipoles(
        self, z: float, term_list: list,
        mixing_matrix: Optional[dict] = None,
        species: Optional[dict] = None,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        gamma: Optional[list] = None
    ) -> dict:
        r"""Convolved power spectrum multipoles of specified terms.

        Parameters
        ----------
        z: float
            Redshift
        term_list: list
            List of terms to compute
        mixing_matrix: dict
            Dictionary containing the mixing matrix
            (used if `species` dictionary is None)
        species: dict, optional
            Dictionary of different samples (main and contaminants), including
            the mixing matrix, effective redshift, fraction, and nuisance
            parameters for each
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        gamma: list, optional
            Distortion factors due to misidentified emissione line (default is
            None)
        Returns
        -------
        multipoles_out: dict
            Convolved power spectrum Legendre multipoles of specified terms
        """
        ells_tot = [0, 2, 4]
        ells = self._ensure_array(ells) if ells is not None else ells_tot

        if species is not None:
            gamma_dict = {
                key: [self.ap_distortion.gamma_tr(val["z"], z),
                      self.ap_distortion.gamma_lo(val["z"], z)]
                for key, val in species.items()}

            multipoles_spec = {}

            for key, val in species.items():
                if "mixing_matrix" not in val:
                    raise ValueError(
                        f"Missing mixing matrix for species '{key}'")

                multipoles_spec[key] = self.convolved_power_term_multipoles(
                    z=val["z"], term_list=term_list,
                    mixing_matrix=val["mixing_matrix"],
                    species=None, ells=ells, use_AP=use_AP,
                    gamma=gamma_dict[key])

            return {f'ell{ell}': np.stack([
                multipoles_spec[key][f"ell{ell}"] * val["fraction"]**2
                for key, val in species.items()]) for ell in ells}

        if mixing_matrix is None:
            raise ValueError(
                "Mixing matrix must be provided when species is None"
            )

        kin_arrays = [mixing_matrix[f"kin{ell}"] for ell in ells_tot]

        multipoles_in = self._input_power_term_multipoles(
            kin_arrays=kin_arrays, z=z, term_list=term_list,
            use_AP=use_AP, gamma=gamma)

        multipoles_out = self._convolve_terms_with_mixing_matrix(
            mixing_matrix, multipoles_in, ells)

        return {"k": mixing_matrix["kout"], **multipoles_out}

    def _UVcutoff(self, k: np.ndarray, kcut: float, pow: float):
        r"""Cutoff of ultraviolet modes

        Parameters
        ----------
        k: np.ndarray
            Input wave modes
        kcut: float
            Cutoff scale
        pow: float
            Index of exponential cutoff

        Returns
        -------
        damping: np.ndarray
            Damping function of UV wave modes
        """
        return np.exp(-((k / kcut) ** pow))

    def two_point_correlation_multipoles(
        self,
        s: np.ndarray,
        ells: Optional[np.ndarray] = None,
        use_AP: Optional[bool] = True,
        logkmin: Optional[float] = -5,
        logkmax: Optional[float] = 2,
        nk: Optional[int] = 2048,
        kcut: Optional[float] = 0.4,
        pow: Optional[float] = 2,
    ) -> dict:
        r"""Two-point correlation function Legendre multipoles.

        Parameters
        ----------
        s: np.ndarray
            Comoving separations
        ells: np.ndarray
            Legendre multipole order
        use_AP: bool
            Flag to switch between with and without AP corrections
        logkmin: float
            Left logarithmic edge of input wave mode array
        logkmax: float
            Right logarithmic edge of input wave mode array
        nk: int
            Number of logarithmic wave mode bins
        kcut: float
            Cutoff scale for exponential damping
        pow: float
            Power index for exponential damping
        Returns
        -------
        multipoles: dict
            Two-point correlation function Legendre multipoles
        """
        if self.spectro_power.NLcode != "COMET":
            raise ValueError(
                "2PCF multipoles can temporarily be retrieved only with COMET"
            )

        ells = self._ensure_array(ells) if ells is not None else np.array([0, 2, 4])
        k_hnkl = np.logspace(logkmin, logkmax, nk)
        pk_multipoles = self.power_multipoles(k=k_hnkl, ells=ells, use_AP=use_AP)
        volume_factor = (k_hnkl**3) / (2 * (np.pi**2))
        xi_multipoles = {}
        for ell in ells:
            y_array = (
                volume_factor
                * pk_multipoles[f"ell{ell}"]
                * self._UVcutoff(k=k_hnkl, kcut=kcut, pow=pow)
                * np.real(1j**ell)
            )
            transformer = fftlog(x=k_hnkl, fx=y_array, nu=2)
            r_grid, transformed_log = transformer.fftlog(ell=ell)
            xi_multipoles[f"ell{ell}"] = np.interp(s, r_grid, transformed_log)
        return xi_multipoles
