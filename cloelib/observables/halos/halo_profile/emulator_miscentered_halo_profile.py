import os
import numpy as np

from cloelib.auxiliary.cluster_emulators import ClusterEmuNet, get_emulator_data
from cloelib.observables.halos.halo_model_properties import HaloModelProperties

from .halo_profile_base import HaloProfileBase
from .bmo_halo_profile import BMOHaloProfile
from .nfw_halo_profile import NFWHaloProfile


class EmulatorMiscenteredHaloProfile(HaloProfileBase):
    r"""
    Miscentered halo profile evaluated via neural-network emulators.

    Two independent six-hidden-layer networks (one for :math:`\Sigma_\mathrm{off}`,
    one for :math:`\Delta\Sigma_\mathrm{off}`) replace the numerical integration
    of the azimuthally averaged miscentered profiles. Each emulator predicts

    .. math::

        \ln\!\left(\frac{\Sigma_\mathrm{off}(R)}{\rho_s}\right)
        \quad\text{or}\quad
        \ln\!\left(\frac{\Delta\Sigma_\mathrm{off}(R)}{\rho_s}\right)

    as a function of the four input features
    :math:`[\log_{10} R,\, \log_{10} R_\mathrm{vir},\, c,\, \sigma_\mathrm{off}]`.

    Parameters
    ----------
    halo_model_properties : HaloModelProperties
        HaloModelProperties object carrying the cosmology.
    overdensity_type : str, optional
        Overdensity type passed to :class:`HaloProfileCore`.  Default ``"vir"``.
    overdensity : int, optional
        Overdensity value passed to :class:`HaloProfileCore`.  Default ``200``.
    z : array_like, optional
        Redshift grid for cosmological calculations.
    zs_max : float, optional
        Maximum source redshift for lensing calculations.
    mean_nz : float, optional
        Mean of the source redshift distribution.
    sigma_nz : float, optional
        Width of the source redshift distribution.
    alpha_nz : float, optional
        Shape parameter of the source redshift distribution.

    Notes
    -----
    The default case used in the implementation as for the BMO halo profile,
    with the emulater trained setting tau_vir = 3.0 within these boundaries
    # Bounds #log10R [cMpc/h] # log10 Rvir [pMpc/h] # c # sigma_off
    lower_bounds = [-4., np.log10(0.15), 0.5, 0.05]
    upper_bounds = [np.log10(30.), np.log10(2.2), 10., 0.8]

    These emulators were trained following Eq. 8 of
    `Johnston et al. 2007 <https://arxiv.org/pdf/0709.1159.pdf>`_ for the
    miscentering PDF and the BMO truncated NFW profile of
    `Baltz et al. 2009 <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.

    Also these Emulators were trained with the ln of the profile, an exp operation
    has to be applied to recover the profiles.
    """

    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        z: np.ndarray = np.linspace(1.0e-5, 6.0 - 1.0e-5, 500),
        zs_max: float = 2.0,
        mean_nz: float = 0.4,
        sigma_nz: float = 0.3,
        alpha_nz: float = 0.4,
    ):
        HaloProfileBase.__init__(
            self,
            halo_model_properties,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
            z=z,
            zs_max=zs_max,
            mean_nz=mean_nz,
            sigma_nz=sigma_nz,
            alpha_nz=alpha_nz,
        )

        self._trunc_fact = None
        self._emu_sigma = None
        self._emu_delta_sigma = None

    def set_weights(
        self,
        sigma_weights: dict | str | None = None,
        delta_sigma_weights: dict | str | None = None,
    ):
        """
        Instantiate and load the two emulator networks from weight dicts,
        falling back to the weights embedded in EmuNetWeights

        Parameters
        ----------
        sigma_weights : dict, str, optional
            Weight dictionary for the :math:`\Sigma_\mathrm{off}` emulator,
            or npz file containing the dictionary. The dictionary must have the
            keys ``fc1_w`` … ``fc6_w``, ``fc1_b`` … ``fc6_b``,
            ``params_min`` (array_like, shape (4,), minima used to normalise the inputs),
            ``params_max`` (array_like, shape (4,), maxima used to normalise the inputs),
            ``profile_model`` (must be ``BMO`` or ``NFW``),
            and ``trunc_fact`` (float, truncation radius in units of the overdensity radius,
            only required if ``profile_model=BMO``).
            Defaults to the weights in `zenodo <https://zenodo.org/records/21380106/files/NN_6hidLwoBN_5e5trainNOSTDwRlg01_bs32_lr1e4red_hs512_3000e_MSELoss_Sigma_1h_off_4parms.npz>`_.
        delta_sigma_weights : dict, str, optional
            Weight dictionary for the :math:`\Delta\Sigma_\mathrm{off}`
            emulator or npz file containing the dictionary. Same keys as ``sigma_weights``.
            Defaults to the weights in `zenodo <https://zenodo.org/records/21380106/files/NN_6hidLwithoutBN_5e5trainNOSTDwRlg01noRescale_bs32_lr1e4red_hs512_3000e_MSELoss_DSigma_1h_off_4parms.npz>`_.
        hidden_size : int, optional
            Hidden-layer width of the emulator networks. Default ``512``.
        """
        # get default data

        zenodo_url = "https://zenodo.org/records/21380106/files"
        datapath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "emulator_data"
        )

        if sigma_weights is None:
            sigma_weights = "NN_6hidLwoBN_5e5trainNOSTDwRlg01_bs32_lr1e4red_hs512_3000e_MSELoss_Sigma_1h_off_4parms.npz"
        if isinstance(sigma_weights, str):
            sigma_weights = get_emulator_data(
                sigma_weights, filepath=datapath, zenodo_url=zenodo_url
            )

        if delta_sigma_weights is None:
            delta_sigma_weights = "NN_6hidLwithoutBN_5e5trainNOSTDwRlg01noRescale_bs32_lr1e4red_hs512_3000e_MSELoss_DSigma_1h_off_4parms.npz"
        if isinstance(delta_sigma_weights, str):
            delta_sigma_weights = get_emulator_data(
                delta_sigma_weights, filepath=datapath, zenodo_url=zenodo_url
            )

        # set up profile

        if (
            sigma_weights["profile_model"].lower()
            != delta_sigma_weights["profile_model"].lower()
        ):
            raise ValueError(
                "Pofile models in sigma_weights and delta_sigma_weights are different!"
            )

        if sigma_weights["profile_model"].lower() == "nfw":
            self._rho_s = NFWHaloProfile._rho_s
        elif sigma_weights["profile_model"].lower() == "bmo":
            if sigma_weights["trunc_fact"] != delta_sigma_weights["trunc_fact"]:
                raise ValueError(
                    "trunc_fact in sigma_weights and delta_sigma_weights are different!"
                )
            self._trunc_fact = sigma_weights["trunc_fact"]
            # tau = R_t / R_s = trunc_fact * c
            self._rho_s = lambda Delta, c: BMOHaloProfile._rho_s(
                Delta, c, tau=self._trunc_fact * c
            )
        else:
            raise ValueError("Pofile model in sigma_weights must be NFW or BMO")

        # set up emulators

        self._emu_sigma = ClusterEmuNet(**sigma_weights["header"])
        self._emu_sigma.load_from_dict(sigma_weights)

        self._emu_delta_sigma = ClusterEmuNet(**delta_sigma_weights["header"])
        self._emu_delta_sigma.load_from_dict(delta_sigma_weights)

    def _emulator_profile(
        self,
        emu: ClusterEmuNet,
        R_mpc: np.ndarray,
        R_vir: np.ndarray,
        densityThreshold: np.ndarray,
        c: float,
        sigma_off: float,
    ) -> np.ndarray:
        r"""
        Run one emulator over the full (z, M, R) grid.

        Parameters
        ----------
        emu : ClusterEmuNet
            The emulator network to evaluate.
        R_mpc : np.ndarray
            Projected radii (Mpc/h), shape ``(1, 1, R.size)``.
        R_vir : np.ndarray
            Virial radii (Mpc/h), shape ``(z.size, M.size, 1)``.
        densityThreshold: np.ndarray
            Threshold density (units : h * Msun / Mpc**2)  with shape (z.size, 1, 1)
        c : float
            Concentration.
        sigma_off : float
            Miscentering scatter (Mpc/h).

        Returns
        -------
        profile : np.ndarray
            Predicted profile (h Msun/pc²), shape ``(z.size, M.size, R.size)``.
        """
        # Broadcast each quantity to (Nz, NM, NR)
        inputs = np.column_stack(
            [
                np.log10(R_mpc).flatten(),
                # convert Rvir (Nz, NM, 1) -> (Nz*NM*NR)
                np.broadcast_to(np.log10(R_vir), R_mpc.shape).flatten(),
                np.full(R_mpc.size, c),
                np.full(R_mpc.size, sigma_off),
            ]
        )  # (Nz*NM*NR, 4)

        # Run emulator (Nz*NM*NR), undo log-scaling and multiply by rho_s
        profile = np.exp(emu.forward(inputs)).reshape(R_mpc.shape) * self._rho_s(
            densityThreshold, c
        )  # Msun h² / Mpc³ · Mpc

        # Convert to h Msun / pc²: 1 Mpc = 1e6 pc  →  1/Mpc² = 1e-12 /pc²
        return profile * 1.0e-12

    def surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered surface mass density profile :math:`\Sigma_\mathrm{off}`.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (units controlled by ``radius_units``).
        z : np.ndarray
            Lens redshifts, shape ``(Nz,)``.
        M : np.ndarray
            Halo masses (Msun / h), shape ``(NM,)``.
        c : float
            Concentration.
        sigma_off : float
            Scatter of the Rayleigh miscentering PDF (Mpc/h),
            see Eq. 8 of `Johnston et al. 2007
            <https://arxiv.org/pdf/0709.1159.pdf>`_.
        radius_units : str, optional
            Unit for the input radii.  Accepted values are
            ``"Mpc/h"``, ``"radians"``, ``"degrees"``, ``"arcmin"``,
            ``"arcsec"``.  Default ``"Mpc/h"``.

        Returns
        -------
        Sigma_off : np.ndarray
            Miscentered surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        if self._emu_sigma is None:
            raise ValueError("Emulator not set up. Run set_weights first")
        Sigma_off = self._emulator_profile(
            self._emu_sigma,
            *self.surface_mass_density_args(R, z, M, radius_units=radius_units),
            c=c,
            sigma_off=sigma_off,
        )

        self.check_profile_shape(R, z, M, Sigma_off)

        return Sigma_off

    def excess_surface_mass_density(
        self,
        R: np.ndarray,
        z: np.ndarray,
        M: np.ndarray,
        c: float,
        sigma_off: float,
        radius_units: str = "Mpc/h",
    ) -> np.ndarray:
        r"""
        Miscentered excess surface mass density profile
        :math:`\Delta\Sigma_\mathrm{off}`.

        Parameters
        ----------
        R : np.ndarray
            Projected radii (units controlled by ``radius_units``).
        z : np.ndarray
            Lens redshifts, shape ``(Nz,)``.
        M : np.ndarray
            Halo masses (Msun / h), shape ``(NM,)``.
        c : float
            Concentration.
        sigma_off : float
            Scatter of the Rayleigh miscentering PDF (Mpc/h),
            see Eq. 8 of `Johnston et al. 2007
            <https://arxiv.org/pdf/0709.1159.pdf>`_.
        radius_units : str, optional
            Unit for the input radii.  Accepted values are
            ``"Mpc/h"``, ``"radians"``, ``"degrees"``, ``"arcmin"``,
            ``"arcsec"``.  Default ``"Mpc/h"``.

        Returns
        -------
        DeltaSigma_off : np.ndarray
            Miscentered excess surface mass density (h Msun / pc²),
            shape ``(Nz, NM, NR)``.
        """
        if self._emu_delta_sigma is None:
            raise ValueError("Emulator not set up. Run set_weights first")
        DeltaSigma_off = self._emulator_profile(
            self._emu_delta_sigma,
            *self.surface_mass_density_args(R, z, M, radius_units=radius_units),
            c=c,
            sigma_off=sigma_off,
        )

        self.check_profile_shape(R, z, M, DeltaSigma_off)

        return DeltaSigma_off
