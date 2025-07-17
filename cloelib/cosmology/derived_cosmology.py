def rho_crit(background, zs: np.ndarray) -> np.ndarray:
    """
    Returns the critical density as a function of redshift.

    Units: Mpc^{-3} Msun

    Args:
        zs (np.ndarray): Redshifts.

    Returns:
        float: Critical density value at the specified redshift.
    """
    h_in_seconds = background.hubble_parameter(zs) / units.MPC_TO_KM
    return 3.0 * h_in_seconds**2.0 / (8.0 * np.pi * units.GRAVITATIONAL_CONSTANT)


def dV_dzdO(background, zs: np.ndarray) -> np.ndarray:
    """
    Returns the volume element per redshit per solid angle
    at the redshift requested.


    Args:
        zs (np.ndarray): Array of redshifts.

    Returns:
        np.ndarray: volume element in Mpc^3 h^{-3}
    """
    return (
        units.SPEED_OF_LIGHT
        / 1.0e3
        * background.comoving_distance(zs) ** 2.0
        * background.hubble_parameter(zs)
    )
