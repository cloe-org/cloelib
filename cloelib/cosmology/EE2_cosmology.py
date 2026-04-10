"""Implementation of nonlinear perturbations using EuclidEmulator2.

## Design

**``EE2NonLinearBoostMixin``** — concrete subclass of the abstract
``NonLinearBoostMixin`` (defined in ``cosmology.py``).  Provides
``nonlinear_boost`` by calling EuclidEmulator2 and building a 2-D
spline interpolator over the extended (z, log k) grid.  Also provides a
custom ``matter_power_spectrum_cb`` that applies the EE2 boost with the
approximation that neutrinos remain linear.

**``EE2NonLinearPerturbations``** — pre-composed class produced by
:func:`~cloelib.cosmology.cosmology.with_nonlinear_boost`.  Equivalent to::

    EE2NonLinearPerturbations = with_nonlinear_boost(
        CAMBLinearPerturbations, EE2NonLinearBoostMixin
    )
"""

# cloelib imports
from cloelib.cosmology.cosmology import (
    Background,
    NonLinearBoostMixin,
    with_nonlinear_boost,
)
from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate

# General imports
import numpy as np
from typing import Sequence

# Cosmology imports
try:
    import euclidemu2

    ee2 = euclidemu2.PyEuclidEmulator()

except ImportError:
    raise ImportError("EuclidEmulator2 could not be imported or initialised.")


class EE2NonLinearBoostMixin(NonLinearBoostMixin):
    """Mixin that provides the EuclidEmulator2 nonlinear boost B(k; z).

    Designed to be composed with a linear perturbations class via
    :func:`~cloelib.cosmology.cosmology.with_nonlinear_boost`::

        from cloelib.cosmology.cosmology import with_nonlinear_boost
        from cloelib.cosmology.camb_cosmology import CAMBLinearPerturbations

        MyEE2Perturbations = with_nonlinear_boost(
            CAMBLinearPerturbations, EE2NonLinearBoostMixin
        )
        pert = MyEE2Perturbations(background, redshifts)

    ``__init__`` reads ``self.background``, ``self.k``, and ``self.z`` from
    the already-initialised linear base class, so it must be called **after**
    the linear class ``__init__``.  The factory :func:`with_nonlinear_boost`
    guarantees this ordering automatically.

    Also provides a custom ``matter_power_spectrum_cb`` implementing the
    approximation that neutrinos remain linear (P_mnν replaced by the linear
    spectrum).
    """

    def __init__(self):
        hubble = self.background.H0 / 100

        params_ee2 = {
            "Omega_b": self.background.Omega_b0,
            "Omega_m": self.background.Omega_m(0),
            "h": hubble,
            "A_s": self.background.As,
            "n_s": self.background.ns,
            "m_ncdm": _set_neutrino_masses(self.background),
            "w0_fld": self.background.w0,
            "wa_fld": self.background.wa,
        }

        ee2_bounds = ee2.bounds

        # At the moment we raise an error when out of range. May decide to extrapolate later
        for key in params_ee2:
            if np.prod(params_ee2[key] - np.array(ee2_bounds[key])) > 0:
                raise ValueError("EE2 out of range.")

        redshift_max = ee2.z_max
        zvals = self.z[self.z <= redshift_max]

        k_emu, boost = ee2.get_boost(params_ee2, zvals)
        k_emu = k_emu * hubble

        boost_arr = np.array([boost[i] for i in range(len(zvals))])

        # Here only the method using interpolators will work in general
        k_out, z_out, boost_out = extend_spectra(
            k_emu,
            zvals,
            boost_arr,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=self.z,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.boost_interp = interpolate.RectBivariateSpline(
            z_out, np.log(k_out), boost_out, kx=1, ky=1
        )
        # Update z to the extended grid used for the boost interpolator
        self.z = z_out

    def nonlinear_boost(self, zs, ks) -> np.ndarray:
        r"""Return the EE2 nonlinear boost B(k, z) = P_NL / P_lin.

        Parameters
        ----------
        zs : np.ndarray
            Redshifts.
        ks : np.ndarray
            Wavenumbers in Mpc\ :sup:`-1`.

        Returns
        -------
        np.ndarray
            Boost array of shape (nz, nk).
        """
        return self.boost_interp(zs, np.log(ks))

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        r"""Compute the CDM+baryons power spectrum with the EE2 boost.

        Uses the approximation that neutrinos remain linear and
        :math:`P_{m\nu}` is replaced by the linear calculation.

        Parameters
        ----------
        zs : np.ndarray
            Redshifts.
        ks : np.ndarray
            Wavenumbers in Mpc\ :sup:`-1`.

        Returns
        -------
        np.ndarray
            CDM+baryons power spectrum at the specified scale and redshift.
        """
        Pcb_L = self._linear_pk_cb(zs, ks)
        Pmm_L = self._linear_pk(zs, ks)
        boost = self.boost_interp(zs, np.log(ks))
        f_cb = (
            self.background.Omega_cdm0 + self.background.Omega_b0
        ) / self.background.Omega_m(0)
        return Pcb_L + (boost - 1) * Pmm_L / f_cb**2


#: EE2 nonlinear perturbations pre-composed via
#: :func:`~cloelib.cosmology.cosmology.with_nonlinear_boost`.
#: Equivalent to
#: ``with_nonlinear_boost(CAMBLinearPerturbations, EE2NonLinearBoostMixin)``.
#: Growth factor, growth rate, and sigma8 are inherited from
#: ``CAMBLinearPerturbations``.
EE2NonLinearPerturbations = with_nonlinear_boost(
    CAMBLinearPerturbations, EE2NonLinearBoostMixin
)


def _set_neutrino_masses(background: Background) -> float:
    r"""Set neutrino masses in the parameters dictionary.

    This method adds neutrino masses to the provided dictionary.
    It also ensures consistency with the background cosmology.
    EE2 only supports a 3 degenerate massive neutrinos, but no error
    thrown, it will just assume take the total mass and use that,
    as this should be a small effect on the boost.

    Parameters
    ----------
    background: Background
        Background class containing cosmology and background quantities

    Returns
    -------
    float
        The total neutrino mass in eV.
    """

    if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
        raise ValueError(
            "EE2 only supports a fixed number of effective"
            f"relativistic species (N_eff=3.044). Found {background.N_eff} "
            "Ensure that N_eff=3.044 in the Background class."
        )
    if isinstance(background.mnu, Sequence) or isinstance(background.mnu, np.ndarray):
        mnu_arg = np.sum(np.asarray(background.mnu))
    else:
        mnu_arg = float(background.mnu)
    # returns the neutrino mass in eV
    return mnu_arg
