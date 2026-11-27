"""Module implementing utils functions for distance calculations."""

import numpy as np

from cloelib.utils import constants


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
    ang_to_rad = {
        "radians": 1,
        "degrees": np.pi / 180.0,
        "arcmin": np.pi / 180.0 / 60.0,
        "arcsec": np.pi / 180.0 / 3600.0,
    }
    _valid_units = ["mpc/h", *ang_to_rad.keys()]
    if units_in.lower() not in _valid_units:
        raise ValueError(f"units_in (={units_in}) must be in {_valid_units}")
    if units_out.lower() not in _valid_units:
        raise ValueError(f"units_out (={units_out}) must be in {_valid_units}")

    if units_in.lower() == units_out.lower():
        return distance

    if units_out.lower() not in ang_to_rad:
        # converting to mpc/h
        theta = distance * ang_to_rad[units_in]  # distance in radians
        out = theta * angular_diameter_distance
    elif units_in.lower() not in ang_to_rad:
        # converting to angular units
        theta = distance / angular_diameter_distance  # distance in radians
        out = theta / ang_to_rad[units_out]
    else:
        out = distance * ang_to_rad[units_in] / ang_to_rad[units_out]

    return out


def isotropic_volume_distance(z, da, hz):
    """Compute isotropic volume distance

    Parameters
    ----------
    z : np.ndarray
        redshift
    da : np.ndarray
        Angular diameter distance
    hz : np.ndarray
        Hubble parameter as a function of redshift.

    Returns
    -------
    np.ndarray
        Isotropic volume distance
    """
    return ((1 + z) ** 2 * da**2 * constants.SPEED_OF_LIGHT * z / hz) ** (1 / 3.0)
