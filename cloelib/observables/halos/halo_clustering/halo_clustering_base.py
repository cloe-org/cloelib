# import jax.numpy as np
import numpy as np
from scipy.integrate import simpson
from scipy.special import sici, spherical_jn

from cloelib.auxiliary.halo_helpers import (
    isotropic_volume_distance,
    photoz_rsd_amplitude,
    photoz_rsd_hexadecapole_correction,
    photoz_rsd_monopole_correction,
    photoz_rsd_quadrupole_correction,
    tophat_window,
)
from cloelib.cosmology.cosmology import Background
from cloelib.observables.halos.halo_model_properties import HaloModelProperties


def _radial_window_f2(x):
    r"""Evaluate the analytic primitive used by the quadrupole shell window.

    Computes

    .. math::

        F_2(x) = \operatorname{Si}(x) - \sin(x),

    where :math:`\operatorname{Si}` is the sine integral. For small
    :math:`|x|`, the function is evaluated with its Taylor expansion to
    avoid cancellation between the two terms.

    Parameters
    ----------
    x : float or np.ndarray
        Dimensionless argument, typically :math:`kr`.

    Returns
    -------
    float or np.ndarray
        Value of :math:`F_2(x)`, with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 0.5
    xs = x[small]
    result[small] = xs**3 * (
        1.0 / 9.0
        + xs**2
        * (
            -1.0 / 150.0
            + xs**2 * (1.0 / 5880.0 + xs**2 * (-1.0 / 408240.0 + xs**2 / 43908480.0))
        )
    )
    xl = x[~small]
    result[~small] = sici(xl)[0] - np.sin(xl)
    return result


def _radial_window_f4(x):
    r"""Evaluate the analytic primitive used by the hexadecapole shell window.

    Computes

    .. math::

        F_4(x) = 3\operatorname{Si}(x)
                 + \sin(x)\left(2 - \frac{15}{x^2}\right)
                 + \frac{15\cos(x)}{x}.

    For small :math:`|x|`, the function is evaluated with its Taylor
    expansion to avoid numerical cancellation between the closed-form
    terms.

    Parameters
    ----------
    x : float or np.ndarray
        Dimensionless argument, typically :math:`kr`.

    Returns
    -------
    float or np.ndarray
        Value of :math:`F_4(x)`, with the same shape as ``x``.

    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1.0
    xs = x[small]
    result[small] = xs**5 * (
        2.0 / 525.0
        + xs**2
        * (
            -1.0 / 6615.0
            + xs**2
            * (1.0 / 374220.0 + xs**2 * (-1.0 / 35675640.0 + xs**2 / 5059454400.0))
        )
    )
    xl = x[~small]
    result[~small] = (
        3.0 * sici(xl)[0] + np.sin(xl) * (2.0 - 15.0 / xl**2) + 15.0 * np.cos(xl) / xl
    )
    return result


class HaloClusteringBase:
    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        background_fid: Background,
    ):
        r"""Parent class for halo-clustering calculation.

        Parameters
        ----------
        halo_model_properties : HaloModelProperties
            An object from the `HaloModelProperties` class.
        background_fid : Background
            Fiducial cosmological background adopted when converting the
            measured two-point correlation function to distances.

        Notes
        -----
        The fiducial background is used only for geometrical
        Alcock--Paczynski corrections, while the model background is taken
        from ``halo_model_properties``.
        """
        self.halo_model_properties = halo_model_properties
        self.background_fid = background_fid

    def radial_shell_window_and_volume(
        self, z: np.ndarray, k: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute the monopole shell window and spherical-shell volume.

        The radial bin edges are first rescaled by the isotropic
        Alcock--Paczynski correction. The shell-averaged monopole window is
        then evaluated analytically from the spherical top-hat window.

        Parameters
        ----------
        z : np.ndarray
            Redshift values at which the geometrical correction is evaluated,
            with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        r : np.ndarray
            Radial-bin edges in :math:`h^{-1}\,\mathrm{Mpc}`, with shape
            ``(n_r + 1,)``.

        Returns
        -------
        shell_window : np.ndarray
            Shell-averaged monopole window with shape ``(n_z, n_r, n_k)``.
        shell_volume : np.ndarray
            Spherical-shell volumes with shape ``(n_z, n_r)``.

        Notes
        -----
        For a shell with corrected edges :math:`r_1` and :math:`r_2`, the
        window is

        .. math::

            W_0(k;\Delta r) =
            \frac{r_2^3 W_{\rm th}(kr_2)-r_1^3 W_{\rm th}(kr_1)}
                 {r_2^3-r_1^3}.
        """

        r_z = (
            self.alcock_paczynski_correction_factor(z)[:, np.newaxis, np.newaxis]
            * r[np.newaxis, :, np.newaxis]
        )  # AP correction (adds a redshift dependence)
        k_r_z = r_z * k[np.newaxis, np.newaxis, :]

        shell_window = np.diff(r_z**3 * tophat_window(k_r_z), axis=1) / (
            np.diff(r_z**3, axis=1)
        )

        shell_volume = 4.0 * np.pi / 3.0 * np.diff(r_z[:, :, 0] ** 3, axis=1)

        return shell_window, shell_volume

    def radial_shell_quadrupole_window_and_volume(
        self, z: np.ndarray, k: np.ndarray, r: np.ndarray, n_quad: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute the quadrupole shell window and spherical-shell volume.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        r : np.ndarray
            Radial-bin edges in :math:`h^{-1}\,\mathrm{Mpc}`, with shape
            ``(n_r + 1,)``.
        n_quad : int, optional
            Retained for API compatibility. The analytic implementation does
            not use numerical quadrature.

        Returns
        -------
        shell_window : np.ndarray
            Shell-averaged quadrupole window with shape
            ``(n_z, n_r, n_k)``.
        shell_volume : np.ndarray
            Spherical-shell volumes with shape ``(n_z, n_r)``.

        Notes
        -----
        This is a convenience wrapper around
        :meth:`radial_shell_multipole_window_and_volume` with ``ell=2``.
        """
        return self.radial_shell_multipole_window_and_volume(z, k, r, 2, n_quad)

    def radial_shell_hexadecapole_window_and_volume(
        self, z: np.ndarray, k: np.ndarray, r: np.ndarray, n_quad: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute the hexadecapole shell window and spherical-shell volume.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        r : np.ndarray
            Radial-bin edges in :math:`h^{-1}\,\mathrm{Mpc}`, with shape
            ``(n_r + 1,)``.
        n_quad : int, optional
            Retained for API compatibility. The analytic implementation does
            not use numerical quadrature.

        Returns
        -------
        shell_window : np.ndarray
            Shell-averaged hexadecapole window with shape
            ``(n_z, n_r, n_k)``.
        shell_volume : np.ndarray
            Spherical-shell volumes with shape ``(n_z, n_r)``.

        Notes
        -----
        This is a convenience wrapper around
        :meth:`radial_shell_multipole_window_and_volume` with ``ell=4``.
        """
        return self.radial_shell_multipole_window_and_volume(z, k, r, 4, n_quad)

    def radial_shell_multipole_window_and_volume(
        self, z: np.ndarray, k: np.ndarray, r: np.ndarray, ell: int, n_quad: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute an analytic shell-averaged multipole window and shell volume.

        The radial-bin edges are rescaled by the isotropic
        Alcock--Paczynski correction and the shell average of the spherical
        Bessel function is evaluated analytically for
        :math:`\ell=0,2,4`.

        Parameters
        ----------
        z : np.ndarray
            Redshift values at which the geometrical correction is evaluated,
            with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        r : np.ndarray
            Radial-bin edges in :math:`h^{-1}\,\mathrm{Mpc}`, with shape
            ``(n_r + 1,)``.
        ell : int
            Multipole order. Must be one of ``0``, ``2``, or ``4``.
        n_quad : int, optional
            Retained for API compatibility. The current analytic
            implementation does not use numerical quadrature.

        Returns
        -------
        shell_window : np.ndarray
            Shell-averaged multipole window with shape
            ``(n_z, n_r, n_k)``.
        shell_volume : np.ndarray
            Spherical-shell volumes with shape ``(n_z, n_r)``.

        Raises
        ------
        ValueError
            If ``ell`` is not one of ``0``, ``2``, or ``4``.

        Notes
        -----
        The window is defined by

        .. math::

            W_\ell(k;\Delta r)
            = \frac{4\pi}{V_{\Delta r}}
              \int_{r_1}^{r_2} dr\,r^2 j_\ell(kr),

        with :math:`V_{\Delta r}=4\pi(r_2^3-r_1^3)/3`.
        """
        if ell not in (0, 2, 4):
            raise ValueError(f"ell (={ell}) must be one of 0, 2, or 4")

        r_z = self.alcock_paczynski_correction_factor(z)[:, np.newaxis] * r
        r1 = r_z[:, :-1, np.newaxis]
        r2 = r_z[:, 1:, np.newaxis]
        ks = np.asarray(k)[np.newaxis, np.newaxis, :]
        delta_r3 = r2**3 - r1**3
        x1 = ks * r1
        x2 = ks * r2
        ks_broadcast = np.broadcast_to(ks, x1.shape)
        delta_r3_broadcast = np.broadcast_to(delta_r3, x1.shape)

        window0 = (r2**3 * tophat_window(x2) - r1**3 * tophat_window(x1)) / delta_r3

        if ell == 0:
            shell_window = window0
        else:
            shell_window = np.empty_like(window0)
            small = np.maximum(np.abs(x1), np.abs(x2)) < 0.25

            coefficients = {
                2: (1.0 / 15.0, -1.0 / 210.0, 1.0 / 7560.0, -1.0 / 498960.0),
                4: (1.0 / 945.0, -1.0 / 20790.0, 1.0 / 1081080.0),
            }[ell]
            series = np.zeros_like(window0)
            for order, coefficient in enumerate(coefficients):
                power = ell + 2 * order
                series += (
                    3.0
                    * coefficient
                    * ks**power
                    * (r2 ** (power + 3) - r1 ** (power + 3))
                    / ((power + 3) * delta_r3)
                )
            shell_window[small] = series[small]

            large = ~small
            if ell == 2:
                f2_x1 = _radial_window_f2(x1[large])
                f2_x2 = _radial_window_f2(x2[large])
                shell_window[large] = (
                    9.0
                    * (f2_x2 - f2_x1)
                    / (ks_broadcast[large] ** 3 * delta_r3_broadcast[large])
                    - window0[large]
                )
            else:
                f4_x1 = _radial_window_f4(x1[large])
                f4_x2 = _radial_window_f4(x2[large])
                window2, _ = self.radial_shell_multipole_window_and_volume(
                    z, k, r, 2, n_quad
                )
                shell_window[large] = (
                    21.0
                    * (f4_x2 - f4_x1)
                    / (2.0 * ks_broadcast[large] ** 3 * delta_r3_broadcast[large])
                    - window2[large]
                )

        shell_volume = 4.0 * np.pi / 3.0 * np.diff(r_z**3, axis=1)
        return shell_window, shell_volume

    # cosmo correction (isotropic AP)
    def alcock_paczynski_correction_factor(self, z: np.ndarray) -> np.ndarray:
        r"""Compute the isotropic Alcock--Paczynski correction factor.

        The correction rescales radial separations measured in the fiducial
        cosmology to those of the model cosmology using the isotropic volume
        distance and the sound horizon at the drag epoch.

        Parameters
        ----------
        z : np.ndarray
            Redshift values at which to evaluate the correction.

        Returns
        -------
        np.ndarray
            Isotropic Alcock--Paczynski correction factor at each redshift,
            with the same shape as ``z``.

        Notes
        -----
        The implemented factor is

        .. math::

            \alpha =
            \frac{D_V(z)}{D_V^{\rm fid}(z)}
            \frac{r_d^{\rm fid}}{r_d},

        where :math:`D_V` is the isotropic volume distance and :math:`r_d` is
        the sound horizon at the drag epoch.
        """

        # units don't matter here, they cancel out

        z[z == 0] = 1e-5

        # isotropic volume distance
        Dv = isotropic_volume_distance(
            z,
            self.halo_model_properties.angular_diameter_distance(z),
            self.halo_model_properties.background.hubble_parameter(z),
        )

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = isotropic_volume_distance(
            z,
            self.background_fid.angular_diameter_distance(z),
            self.background_fid.hubble_parameter(z),
        )

        return (Dv / Dv_fid) * (
            self.background_fid.rdrag / self.halo_model_properties.background.rdrag
        )

    def photoz_rsd_halo_correction(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the photo-z and RSD halo correction for the monopole.

        The correction combines the analytic dispersion-model monopole
        coefficients with the effective halo bias and the linear growth rate.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        z_obs_scatter : np.ndarray
            Observed redshift scatter :math:`\sigma_z`. Its first dimension
            must correspond to redshift.
        b_eff : np.ndarray
            Effective linear halo bias. Must have the same shape as
            ``z_obs_scatter``.

        Returns
        -------
        np.ndarray
            Monopole halo correction with shape
            ``(n_z, n_k, ...)``, where the trailing dimensions are those of
            ``z_obs_scatter`` after the redshift axis.

        Raises
        ------
        ValueError
            If ``z_obs_scatter`` and ``b_eff`` do not have identical shapes.

        Notes
        -----
        The returned quantity is

        .. math::

            b_{\rm eff}^2 A_0
            + b_{\rm eff} f B_0
            + f^2 C_0,

        where :math:`A_0`, :math:`B_0`, and :math:`C_0` are the analytic
        dispersion-model coefficients and :math:`f=\Omega_{cb}^{0.55}`.
        """
        if z_obs_scatter.shape != b_eff.shape:
            raise ValueError(
                f"Shape of z_obs_scatter {z_obs_scatter.shape} must be"
                f" the same as b_eff {b_eff.shape}"
            )
        # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
        # rsd corrections (z, k, ...)
        photoz_corr0, photoz_corr1, photoz_corr2 = photoz_rsd_monopole_correction(
            self.halo_model_properties.background, z, k, z_obs_scatter
        )

        # halo correction (z, k, ...)
        b_eff_reshaped = b_eff[:, np.newaxis]  # add k axis in position 1
        photoz_halo_corr = (
            photoz_corr0 * b_eff_reshaped**2
            + photoz_corr1 * b_eff_reshaped
            + photoz_corr2
        )

        return photoz_halo_corr

    def photoz_rsd_halo_quadrupole_correction(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the photo-z and RSD halo correction for the quadrupole.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        z_obs_scatter : np.ndarray
            Observed redshift scatter :math:`\sigma_z`. Its first dimension
            must correspond to redshift.
        b_eff : np.ndarray
            Effective linear halo bias. Must have the same shape as
            ``z_obs_scatter``.

        Returns
        -------
        np.ndarray
            Quadrupole halo correction with shape ``(n_z, n_k, ...)``.

        Raises
        ------
        ValueError
            If ``z_obs_scatter`` and ``b_eff`` do not have identical shapes.

        Notes
        -----
        The returned quantity is

        .. math::

            b_{\rm eff}^2 A_2
            + b_{\rm eff} f B_2
            + f^2 C_2,

        using the analytic :math:`\ell=2` dispersion-model coefficients.
        """
        if z_obs_scatter.shape != b_eff.shape:
            raise ValueError(
                f"Shape of z_obs_scatter {z_obs_scatter.shape} must be"
                f" the same as b_eff {b_eff.shape}"
            )

        corr0, corr1, corr2 = photoz_rsd_quadrupole_correction(
            self.halo_model_properties.background, z, k, z_obs_scatter
        )
        bias = b_eff[:, np.newaxis]
        return corr0 * bias**2 + corr1 * bias + corr2

    def photoz_rsd_halo_hexadecapole_correction(self, z, k, z_obs_scatter, b_eff):
        r"""Compute the photo-z and RSD halo correction for the hexadecapole.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        z_obs_scatter : np.ndarray
            Observed redshift scatter :math:`\sigma_z`. Its first dimension
            must correspond to redshift.
        b_eff : np.ndarray
            Effective linear halo bias. Must have the same shape as
            ``z_obs_scatter``.

        Returns
        -------
        np.ndarray
            Hexadecapole halo correction with shape ``(n_z, n_k, ...)``.

        Raises
        ------
        ValueError
            If ``z_obs_scatter`` and ``b_eff`` do not have identical shapes.

        Notes
        -----
        The returned quantity is

        .. math::

            b_{\rm eff}^2 A_4
            + b_{\rm eff} f B_4
            + f^2 C_4,

        using the analytic :math:`\ell=4` dispersion-model coefficients.
        """
        if z_obs_scatter.shape != b_eff.shape:
            raise ValueError(
                f"Shape of z_obs_scatter {z_obs_scatter.shape} must be"
                f" the same as b_eff {b_eff.shape}"
            )

        corr0, corr1, corr2 = photoz_rsd_hexadecapole_correction(
            self.halo_model_properties.background, z, k, z_obs_scatter
        )
        bias = b_eff[:, np.newaxis]
        return corr0 * bias**2 + corr1 * bias + corr2

    def photoz_rsd_halo_amplitude(self, z, k, z_obs_scatter, b_eff, mu):
        r"""Compute the redshift-space halo amplitude at fixed line-of-sight angle.

        Parameters
        ----------
        z : np.ndarray
            Redshift values, with shape ``(n_z,)``.
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        z_obs_scatter : np.ndarray
            Observed redshift scatter :math:`\sigma_z`. Its first dimension
            must correspond to redshift.
        b_eff : np.ndarray
            Effective linear halo bias. Must have the same shape as
            ``z_obs_scatter``.
        mu : float or np.ndarray
            Cosine of the angle between the wavevector and the line of sight.

        Returns
        -------
        np.ndarray
            Damped redshift-space halo amplitude with shape
            ``(n_z, n_k, ...)``.

        Raises
        ------
        ValueError
            If ``z_obs_scatter`` and ``b_eff`` do not have identical shapes.

        Notes
        -----
        The returned amplitude is

        .. math::

            \left(b_{\rm eff}+f\mu^2\right)
            \exp\left[-\frac{1}{2}(k\sigma_r\mu)^2\right],

        whose square gives the Kaiser-plus-Gaussian-damping factor entering
        the anisotropic dispersion-model power spectrum.
        """
        if z_obs_scatter.shape != b_eff.shape:
            raise ValueError(
                f"Shape of z_obs_scatter {z_obs_scatter.shape} must be"
                f" the same as b_eff {b_eff.shape}"
            )
        return photoz_rsd_amplitude(
            self.halo_model_properties.background,
            z,
            k,
            z_obs_scatter,
            b_eff,
            mu,
        )

    # IR resummation of the bao wiggles in the Pk
    # not in use currently
    def Pk_IR_func(self, k: np.array, Pk: np.ndarray) -> np.ndarray:
        r"""Apply first-order infrared resummation to BAO wiggles.

        The linear matter power spectrum is decomposed into smooth and
        oscillatory components using an Eisenstein--Hu reference spectrum and
        Gaussian filtering in :math:`\log_{10}k`. The oscillatory component
        is then exponentially damped.

        Parameters
        ----------
        k : np.ndarray
            Wavenumbers in :math:`h\,\mathrm{Mpc}^{-1}`, with shape
            ``(n_k,)``.
        Pk : np.ndarray
            Linear matter power spectrum in
            :math:`(h^{-1}\,\mathrm{Mpc})^3`, with shape
            ``(n_z, n_k)``.

        Returns
        -------
        Pk_IR : np.ndarray
            Infrared-resummed matter power spectrum in
            :math:`(h^{-1}\,\mathrm{Mpc})^3`, with the same shape as ``Pk``.

        Notes
        -----
        The implementation constructs a no-wiggle component from the
        Eisenstein & Hu (1998) transfer-function approximation and returns

        .. math::

            P_{\rm IR}(k)
            = P_{\rm nw}(k)
              + \exp[-k^2\Sigma^2]\,P_{\rm w}(k).

        The input ``k`` array is temporarily rescaled in place from
        :math:`h\,\mathrm{Mpc}^{-1}` to :math:`\mathrm{Mpc}^{-1}` and is
        rescaled back before returning.
        """

        ns = self.halo_model_properties.background.ns
        h = self.halo_model_properties.background.h
        Obh2 = self.halo_model_properties.Omega_b_0 * h**2
        Omh2 = self.halo_model_properties.Omega_m_0 * h**2
        Tcmb = 2.73

        k *= h  #  1/Mpc
        s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1.0 + 10.0 * (Obh2) ** 0.75)
        Gamma = Omh2 / h
        AG = (
            1.0
            - 0.328 * np.log(431.0 * Omh2) * Obh2 / Omh2
            + 0.38 * np.log(22.3 * Omh2) * (Obh2 / Omh2) ** 2
        )
        Gamma = Gamma * (AG + (1.0 - AG) / (1.0 + (0.43 * k * s) ** 4))
        Theta = Tcmb / 2.7
        q = k * Theta**2 / Gamma / h
        L0 = np.log(2.0 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T0 = L0 / (L0 + C0 * q * q)
        T0 /= T0[0]
        P_EH = k**ns * T0**2
        P_EH = (P_EH[:, None] * (Pk[:, 0] / P_EH[0])).T
        k /= h  # h/Mpc again

        lamb = 0.25
        kS = 0.2
        lOsc = 102.707
        qlog = np.log10(k)
        dqlog = qlog[1] - qlog[0]

        # Gaussian filtering for Pnw and Wiggle-smooth split
        Pnw = (
            P_EH
            * np.array(
                [
                    dqlog
                    / np.sqrt(2.0 * np.pi * lamb**2)
                    * (
                        np.sum(
                            np.e
                            ** (
                                -0.5
                                * (klog - qlog[(abs(qlog - klog) < 4.0 * lamb)]) ** 2
                                / lamb**2
                            )
                            * Pk[:, (abs(qlog - klog) < 4.0 * lamb)]
                            / P_EH[:, (abs(qlog - klog) < 4.0 * lamb)],
                            axis=1,
                        )
                    )
                    for klog in qlog
                ]
            ).T
        )

        Pw = Pk - Pnw

        # Sigma2 as integral up to 0.2
        icut = k <= kS
        kcut = k[icut]
        Pnwcut = Pnw[:, icut]
        kosc = 1.0 / lOsc
        norm = 1.0 / (6.0 * np.pi**2)
        Sigma2 = norm * simpson(
            Pnwcut
            * (1.0 - spherical_jn(0, kcut / kosc) + 2.0 * spherical_jn(2, kcut / kosc)),
            x=kcut,
        )

        # comput final power spectrum
        Pk_IR = Pnw + np.e ** (-(k**2) * Sigma2[:, None]) * Pw

        return Pk_IR
