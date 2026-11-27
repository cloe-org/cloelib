"""Background cosmology for the coupled models (for the moment exponential conformal coupling)"""

from typing import Optional, Sequence, Union

import numpy as np

from cloelib.cosmology.class_cosmology import CLASSBackground


class DARKINBackground(CLASSBackground):
    """Configure a DARKIN build of classy through the existing CLASS interface.

    Here ``Omega_cdm0`` denotes the combined present-day ordinary-CDM and
    coupled CDM (QCDM) density fraction. A small ordinary-CDM component
    sets the synchronous gauge reference species; the remaining density is
    assigned to QCDM.

    The scalar replaces the dark energy fluid. The inherited ``w0`` and ``wa``
    arguments remain for CLOE interface compatibility but do not set its
    equation of state. Other potentials and coupling sectors are not allowed
    in this initial implementation - to be extended.

    We also add the particular DARKIN parameters ``scf_beta`` and
    ``scf_lambda`` for the conformal coupling and exponential potential.
    The scalar initial conditions are set by ``scf_phi ini`` and
    ``scf_phi_prime_ini``. The potential scale is determined by shooting on
    ``scf_V0``. ``residual_omega_cdm`` sets the residual ordinary-CDM density.
    """

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu: Union[float, Sequence[float], np.ndarray],
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        scf_beta: float,
        scf_lambda: float,
        N_ur: Optional[float] = None,
        alpha_s: float = 0.0,
        scf_phi_ini: float = 1.0,
        scf_phi_prime_ini: float = 0.0,
        residual_omega_cdm: float = 1e-10,
    ) -> None:
        """Initialize the conformal scalar background.

        Standard cosmological inputs use CLASSBackground conventions, except
        that Omega_cdm0 includes QCDM as described above. scf_beta and
        scf_lambda are the coupling strength and exponential-potential slope.
        Scalar initial conditions use DARKIN's input conventions.
        The potential normalisation is determined by shooting on scf_V0.
        """
        if (
            not np.all(
                np.isfinite(
                    [
                        H0,
                        Omega_cdm0,
                        scf_beta,
                        scf_lambda,
                        scf_phi_ini,
                        scf_phi_prime_ini,
                        residual_omega_cdm,
                    ]
                )
            )
            or H0 <= 0
        ):
            raise ValueError("DARKIN inputs must be finite and H0 must be positive.")

        omega_dm = Omega_cdm0 * (H0 / 100.0) ** 2
        if not 0 < residual_omega_cdm < omega_dm:
            raise ValueError("Require 0 < residual_omega_cdm < total omega_dm.")

        self.scf_beta = scf_beta
        self.scf_lambda = scf_lambda
        self.scf_phi_ini = scf_phi_ini
        self.scf_phi_prime_ini = scf_phi_prime_ini
        self.residual_omega_cdm = residual_omega_cdm

        # ET: Reuse common CLASS handling while configuring the scalar sector.
        super().__init__(
            H0=H0,
            Omega_b0=Omega_b0,
            Omega_cdm0=Omega_cdm0,
            Omega_k0=Omega_k0,
            As=As,
            ns=ns,
            mnu=mnu,
            w0=w0,
            wa=wa,
            gamma_MG=gamma_MG,
            N_mnu=N_mnu,
            N_ur=N_ur,
            alpha_s=alpha_s,
            omega_cdm=residual_omega_cdm,
            omega_qcdm=omega_dm - residual_omega_cdm,
            w0_fld=None,
            wa_fld=None,
            use_ppf=None,
            Omega_Lambda=0.0,
            Omega_fld=0.0,
            Omega_scf=-1.0,
            scf_potential="exp",
            scf_coupling_type="conformal",
            scf_beta=scf_beta,
            scf_lambda=scf_lambda,
            scf_V0=1.0,
            scf_C0=1.0,
            scf_shooting_target="scf_V0",
            attractor_ic_scf="no",
            scf_phi_ini=scf_phi_ini,
            scf_phi_prime_ini=scf_phi_prime_ini,
        )

    def Omega_cb(self, zs):
        """Return baryon + ordinary-CDM + QCDM density fractions, without neutrinos.

        Interpolate the solver's background table within its redshift range.
        """
        # ET: Om_cdm alone excludes QCDM; store the complete cold-matter fraction.
        if not hasattr(self, "_cb_redshifts"):
            table = self.results.get_background()
            required = {"z", "(.)rho_b", "(.)rho_cdm", "(.)rho_qcdm", "(.)rho_crit"}
            if not required.issubset(table):
                raise RuntimeError(
                    "The loaded classy build must give DARKIN QCDM densities."
                )
            order = np.argsort(table["z"])
            self._cb_redshifts = np.asarray(table["z"])[order]
            self._cb_fractions = (
                (
                    np.asarray(table["(.)rho_b"])
                    + np.asarray(table["(.)rho_cdm"])
                    + np.asarray(table["(.)rho_qcdm"])
                )
                / np.asarray(table["(.)rho_crit"])
            )[order]

        z = np.asarray(zs)
        if (
            not np.all(np.isfinite(z))
            or np.any(z < self._cb_redshifts[0])
            or np.any(z > self._cb_redshifts[-1])
        ):
            raise ValueError(
                "Requested redshifts lie outside the DARKIN background table."
            )
        result = np.interp(z, self._cb_redshifts, self._cb_fractions)
        return float(result) if z.ndim == 0 else result
