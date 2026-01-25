"""Module implementing auxiliary functions"""

import numpy as np
from astropy import units as ap_units


def convert_to_Delta_crit(
    overdensity_type, overdensity=200, background=None, z=0.0, nonu=False
):
    r"""Critical overdensity factor.

    Converts the input overdensity factor into a critical one.

    Parameters
    ----------
    overdensity_type : str
        Overdensity definition for halo mass calculation. Must be one of:
        - "crit": Relative to critical density of the universe.
        - "mean": Relative to mean matter density.
        - "vir": Virial overdensity from spherical collapse.
    overdensity : int, optional
        Value of the overdensity. Effective for non-virial overdensities.
        Example: If it equals 200, halos are defined as regions with density
        200 times the chosen reference (`crit` or `mean`).
    background : Background
        Background object.
    z: float or np.ndarray
        Redshift.
    nonu : bool, optional
        If `True`, massive neutrinos are excluded from the density parameter
        summation.

    Returns
    -------
    overdensity: float or np.ndarray
        The overdensity factor which needs
        to be multiplied to the critical
        density in order to define an overdensity.

    Notes
    -----
    The function is returned for :math:`\rm \rho_c` in a density definition
    at a given redshift. The function returns :math:`\rm \Delta` for the
    critical density of the universe, :math:`\rm \Delta \Omega_{m}` for
    the mean matter density of the universe, :math:`\rm \Delta` determined
    by `Bryan & Norman 1998
    <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`_ Equation 6 for
    the virial density.
    """
    if overdensity_type not in ["crit", "mean", "vir"]:
        raise ValueError("Invalid overdensity definition, %s." % overdensity_type)

    if overdensity_type in ["mean", "vir"]:
        if nonu:
            Omega_m = background.Omega_m_cb(z)
        else:
            Omega_m = background.Omega_m(z)

    if overdensity_type == "crit":
        return overdensity

    elif overdensity_type == "mean":
        return overdensity * Omega_m

    elif overdensity_type == "vir":
        x = Omega_m - 1.0
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def convert_distance(distance, units_in, units_out, angular_diameter_distance=None):
    r"""Convert distances

    Parameters
    ----------
    distance: np.ndarray
        Input projected distances
    units_in: str
        Unit for the input projected distance. Accepted values are:
        "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
    units_out: str
        Unit for the output projected distance. Accepted values are:
        "Mpc/h", "radians", "degrees", "arcmin", "arcsec".
    angular_diameter_distance: float, np.ndarray
        Angular diameter distance (units: Mpc/h) to be used for converting
        between angular and physical units. If array, it
        should be in the shape (z.size, 1).

    Returns
    -------
    np.ndarray
        Distance in output units. If z is array and physical to
        angular conversion used, output shape is (z.size, distance.size).
    """
    angular_units_dict = {
        "radians": ap_units.rad,
        "degrees": ap_units.deg,
        "arcmin": ap_units.arcmin,
        "arcsec": ap_units.arcsec,
    }
    _valid_units = ["mpc/h", *angular_units_dict.keys()]
    if units_in.lower() not in _valid_units:
        raise ValueError(f"units_in (={units_in}) must be in {_valid_units}")
    if units_out.lower() not in _valid_units:
        raise ValueError(f"units_out (={units_out}) must be in {_valid_units}")

    if units_in.lower() == units_out.lower():
        return distance

    if units_out.lower() not in angular_units_dict:
        # converting to mpc/h
        theta = (
            (distance * angular_units_dict[units_in]).to(ap_units.rad).value
        )  # distance in radians
        out = theta * angular_diameter_distance
    elif units_in.lower() not in angular_units_dict:
        # converting to angular units
        theta = distance / angular_diameter_distance  # distance in radians
        out = (theta * ap_units.rad).to(angular_units_dict[units_out]).value
    else:
        out = (
            (distance * angular_units_dict[units_in])
            .to(angular_units_dict[units_out])
            .value
        )

    return out
