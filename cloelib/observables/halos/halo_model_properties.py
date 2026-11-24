import numpy as np
from scipy import interpolate
from scipy.integrate import quad_vec
from scipy.special import j0, j1
import importlib

from cloelib.auxiliary.halo_helpers import convert_distance

from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations

# Get classes for verification of units argument

modules_and_functions = {
    "cloelib.cosmology.class_cosmology": (
        "CLASSLinearPerturbations",
        "CLASSNonLinearPerturbations",
    ),
    "cloelib.cosmology.camb_cosmology": (
        "CAMBLinearPerturbations",
        "CAMBNonLinearPerturbations",
    ),
    "cloelib.cosmology.jax_cosmology": (
        "JAXLinearPerturbations",
        "JAXNonLinearPerturbations",
    ),
    "cloelib.cosmology.HMcode2020Emu_cosmology": (
        "HMemuLinearPerturbations",
        "HMemuNonLinearPerturbations",
    ),
    "cloelib.cosmology.mochi_class_cosmology": (
        "mochiCLASSLinearPerturbations",
        "mochiCLASSNonLinearPerturbations",
    ),
    "cloelib.cosmology.hi_class_cosmology": (
        "hi_classLinearPerturbations",
        "hi_classNonLinearPerturbations",
    ),
    "cloelib.cosmology.mgclass_cosmology": (
        "MGCLASSLinearPerturbations",
        "MGCLASSNonLinearPerturbations",
    ),
}

_COSMOLOGY_TYPES_WITH_UNITS = ()
for module_name, cosmo_classes in modules_and_functions.items():
    try:
        module = importlib.import_module(module_name)

        for cosmo_class in cosmo_classes:
            _COSMOLOGY_TYPES_WITH_UNITS = (
                *_COSMOLOGY_TYPES_WITH_UNITS,
                getattr(module, cosmo_class),
            )
    except ImportError:
        pass


def _bessel_j2(x):
    """Bessel function j2"""
    return 2.0 / x * j1(x) - j0(x)


class HaloModelProperties:
    def __init__(
        self,
        perturbations: Perturbations,
        interpolate_pk: bool = True,
        interpolate_da: bool = True,
        z=np.linspace(1.0e-5, 2.0 - 1.0e-5, 100),
        k=np.geomspace(1e-4, 10, 500),
    ):
        r"""Auxiliary class computing quantities used in galaxy cluster models.

        Initialize the class with given perturbations and overdensity definition.
        If requested, cosmological functions (e.g., the matter power spectrum)
        are interpolated.

        Parameters
        ----------
        perturbations : Perturbations
            An object from the `LinearPerturbations` class containing cosmological
            perturbation data (e.g., power spectrum, growth function).
        interpolate_pk : bool, optional
            If true, the class interpolates the matter power spectrum.
            A default interpolation is set when class is instanciated with
            interpolate_pk=True. For a more customized interpolation, check
            the set_matter_power_spectrum_interpolation function.
        interpolate_da : bool, optional
            If true, the class interpolates the angular diameter distance.
            A default interpolation is set when class is instanciated with
            interpolate_da=True. For a more customized interpolation, check
            the set_angular_diameter_distance_interpolation function.

        Notes
        ----------
        In the current implementation, the matter power spectrum never includes
        the contribution of massive neutrinos. Halo mass function, halo bias, and
        2-halo profile models implemented in this subpackage require
        cold dark matter + baryons (no neutrinos) power spectra.
        """
        self.perturbations = perturbations
        self.k = k

        # Interpolators
        self.Pk_interp_cb = None
        self.da_interp = None

        # set P(k) interpolation usage
        if interpolate_pk:
            self.set_matter_power_spectrum_interpolation(z, self.k)
        self.interpolate_pk = interpolate_pk

        # set angular diameter distance interpolation usage
        if interpolate_da:
            self.set_angular_diameter_distance_interpolation(z)
        self.interpolate_da = interpolate_da

        # Density parameters at z=0
        self.Omega_m_0 = self.background.Omega_m(0.0)
        self.Omega_cb_0 = self.background.Omega_cb(0.0)
        self.Omega_b_0 = self.background.Omega_b(0.0)

    @property
    def background(self):
        r"""Returns the Background class instance"""
        return self.perturbations.background

    def angular_diameter_distance(self, z):
        """
        Return the angular diameter distance as a function of redshift.

        Parameters
        ----------
        z : np.ndarray
            Array of redshifts.

        Returns
        -------
        np.ndarray
            Angular diameter distance values.
        """
        if self.interpolate_da:
            _angular_diameter_distance_func = self.da_interp
        else:
            _angular_diameter_distance_func = self.background.angular_diameter_distance
        if _angular_diameter_distance_func is None:
            raise ValueError(
                "Cosmology not instanciated, matter power spectrum function is None!"
            )
        return _angular_diameter_distance_func(z)

    def matter_power_spectrum_cb(self, z, k):
        r"""Computes the non interpolated matter power spectrum.

        This function computes the cold dark matter + baryons power spectrum,
        not including the massive neutrino contribution.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
            Wavenumber where W(kR) is evaluated.
            Units: h Mpc^{-1}

        Returns
        -------
        float or np.ndarray
            Matter power spectrum.
        """
        if self.interpolate_pk:
            _matter_power_spectrum_cb_func = self.Pk_interp_cb
        else:
            _matter_power_spectrum_cb_func = self._matter_power_spectrum_cb_exact
        if _matter_power_spectrum_cb_func is None:
            raise ValueError(
                "Cosmology not instanciated, matter power spectrum function is None!"
            )
        return _matter_power_spectrum_cb_func(z, k)

    def _matter_power_spectrum_cb_exact(self, z, k):
        r"""Computes the non interpolated matter power spectrum.

        Just as wrapper of self.perturbations.matter_power_spectrum_cb
        """
        _kwargs = {}
        if isinstance(
            self.perturbations,
            _COSMOLOGY_TYPES_WITH_UNITS,
        ):
            _kwargs = {
                "hubble_units": True,
                "k_hunit": True,
            }
        return self.perturbations.matter_power_spectrum_cb(z, k, **_kwargs)

    def set_matter_power_spectrum_interpolation(self, z, k):
        r"""Create internal interpolation of matter power spectrum.

        This function interpolates the cold dark matter + baryons power spectrum,
        not including the massive neutrino contribution.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        k: float or np.ndarray
            Wavenumber where W(kR) is evaluated.
            Units: h Mpc^{-1}
        """
        # Power spectrum interpolation
        self.Pk_interp_cb = interpolate.RectBivariateSpline(
            z,
            k,
            self._matter_power_spectrum_cb_exact(z, k),
        )

    def set_angular_diameter_distance_interpolation(self, z):
        r"""Create internal interpolation of angular diameter distance.

        Parameters
        ----------
        z: float or np.ndarray
            Redshift.
        """
        self.da_interp = interpolate.InterpolatedUnivariateSpline(
            x=z,
            y=self.background.angular_diameter_distance(z),
            ext=2,
        )

    def _generic_mass_density_2h(self, R, z, bessel_function, radius_units="Mpc/h"):
        r"""
        Surface or excess surface 2-halo density profile.

        Computes either the cosmological unbiased surface or excess surface
        (depending on the input Bessel function) 2-halo density profile.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        bessel_function: function
            Bessel function that goes in the integrand with the power spectrum.
            Used to return the surface density or the excess surface density.
            It should take (ll*theta) as input.

        Returns
        -------
        profile: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        # Calculate all redshift base quantities and shape them (nz, 1)
        z_plus_1 = np.asarray(z)[:, np.newaxis] + 1.0
        D_A = self.angular_diameter_distance(z)[:, np.newaxis] * self.background.h
        rho_m = (
            self.background.Omega_m(z)
            * derived_cosmology.rho_crit(self.background, z)
            / self.background.h**2
        )[:, np.newaxis]

        ## 1. Get radial distance in radians
        theta = convert_distance(R, radius_units, "radians", D_A)
        if radius_units.lower() != "mpc/h":
            # in this case, theta was missing z dimension
            theta = theta[np.newaxis, :]

        ## 2. Integrand function
        def integrand(kl):
            ll = kl * z_plus_1 * D_A
            Pk_vals = self.matter_power_spectrum_cb(z, kl)
            return bessel_function(ll * theta) * ll * Pk_vals

        ## 3. Integration
        two_point_corr = (
            quad_vec(integrand, self.k.min(), self.k.max(), epsrel=1e-1)[0]
            * z_plus_1
            * D_A
        )

        # Final strictly 2D calculation
        profile = (1.0e-12 * rho_m * two_point_corr) / (
            2.0 * np.pi * z_plus_1**3.0 * D_A**2.0
        )

        return profile

    def surface_mass_density_2h(self, R, z, radius_units="Mpc/h"):
        r"""
        Surface 2-halo matter density profile.

        Computes the cosmological unbiased surface 2-halo density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, bessel_function=j0, radius_units=radius_units
        )

    def excess_surface_mass_density_2h(self, R, z, radius_units="Mpc/h"):
        r"""
        Excess surface 2-halo matter density profile.

        Computes the cosmological unbiased excess surface 2-halo
        density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            2-halo surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, R.size).
        """
        return self._generic_mass_density_2h(
            R, z, bessel_function=_bessel_j2, radius_units=radius_units
        )
