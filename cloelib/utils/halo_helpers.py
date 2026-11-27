"""Module implementing utils functions for halos"""

import numpy as np
from scipy.special import erf
from cloelib.utils.constants import SPEED_OF_LIGHT_KM_S


def convert_to_Delta_crit(overdensity_type, overdensity=200, background=None, z=0.0):
    r"""Critical overdensity factor.

    Converts the input overdensity factor into a critical one.
    The contribution from massive neutrinos is not included in the matter density parameter.

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

    if overdensity_type == "crit":
        return overdensity

    elif overdensity_type == "mean":
        return overdensity * background.Omega_cb(z)

    elif overdensity_type == "vir":
        x = background.Omega_cb(z) - 1.0
        return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def photoz_rsd_correction(
    background,
    z: np.ndarray,
    k: np.ndarray,
    z_obs_scatter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the correction that accounts for photo-z uncertainty and RSD (Kaiser effect),
    from `(Kaiser (1987)) <(https://doi.org/10.1093/mnras/227.1.1>`_.

    Parameters
    ----------
    background: Background
        Background class containing cosmology
    k:  np.ndarray
        wavenumber
    z:  np.ndarray
        redshift
    z_obs_scatter: float, numpy.ndarray
        Observed redshift scatter. If array, first dimension must be z.

    Returns
    -------
    corr0, corr1, corr2: np.ndarray
        Correction terms to the power spectrum monopole
        Shape (z.size, k.size, other dimensions of z_obs_scatter)
    """
    ks = np.atleast_1d(k)
    zs = np.atleast_1d(z)
    z_obs_scatter_arr = np.array(z_obs_scatter)

    # growth rate and scaled k, shape (z.size, k.size)
    f_gr = (background.Omega_cb(zs) ** 0.55)[:, np.newaxis]
    ks_z = (
        ks[np.newaxis, :]
        * SPEED_OF_LIGHT_KM_S
        / background.hubble_parameter(zs)[:, np.newaxis]
        * (background.H0 / 100)
    )

    # check if z_obs_scatter has more dimensions
    ndim_z_obs_scatter = len(z_obs_scatter_arr.shape)
    if ndim_z_obs_scatter > 1:
        # if it does, add them to f_gr, ks_z
        extra_axes = tuple(range(2, ndim_z_obs_scatter + 1))
        f_gr = np.expand_dims(f_gr, axis=extra_axes)
        ks_z = np.expand_dims(ks_z, axis=extra_axes)
    if ndim_z_obs_scatter > 0:
        # if z_obs_scatter is array, add k dimention in 2nd place
        z_obs_scatter_arr = z_obs_scatter_arr[:, np.newaxis, ...]

    # multiply by scatter
    ks_z = ks_z * z_obs_scatter_arr

    erf_ks = erf(ks_z)

    corr0 = np.sqrt(np.pi) / (2 * ks_z) * erf_ks
    corr1 = f_gr / ks_z**3 * (np.sqrt(np.pi) / 2 * erf_ks - ks_z * np.exp(-(ks_z**2)))
    corr2 = (
        f_gr**2
        / ks_z**5
        * (
            3 * np.sqrt(np.pi) / 8 * erf_ks
            - ks_z / 4 * (2 * ks_z**2 + 3) * np.exp(-(ks_z**2))
        )
    )

    # correct for numerical inaccuracy
    # note: for jax, use corr1 = corr1.at[idx].set(2 / 3.0)
    idx = erf_ks < 0.02
    corr1[idx] = 2 / 3.0
    corr2[idx] = 1 / 5.0

    return corr0, corr1, corr2


def tabulated_return(reference_table, func, func_kwargs):
    """Check if arguments given are the same from the reference table.
    If true, returns the tabulated value. Otherwise, recomputes the output
    value and stores it (and inputs) back in reference table.

    Parameters
    ----------
    reference_table: dict
        Dictionary with the tabulated function. It must contain a `inputs`
        key with the dictionary of arguments or `function` and a `values`
        key with the computed value of said function with those arguments.
    func: function
        Function to be tabulated
    func_kwargs: dict
        Dictionary with named arguments for the function.

    Returns
    -------
    Output of `func(**func_kwargs)`
    """

    # Check if names of parameters given are the same as in reference_table
    _set_tab = set(reference_table["inputs"].keys())
    _set_inp = set(func_kwargs.keys())
    if _set_tab != _set_inp:
        raise ValueError(
            f"Bad parameters were passed. Expected {_set_tab}, got {_set_inp}."
        )

    # Check if func_kwargs are the tabluated values
    _tabuleted_input = True
    if any(value is None for value in reference_table["inputs"].values()):
        _tabuleted_input = False
    else:
        for name, ref_val in reference_table["inputs"].items():
            test_val = func_kwargs[name]
            if ref_val.shape != test_val.shape:
                _tabuleted_input = False
                break
            if (ref_val != test_val).any():
                _tabuleted_input = False
                break

    if not _tabuleted_input:
        reference_table["inputs"].update(func_kwargs)
        reference_table["values"] = func(**func_kwargs)

    return reference_table["values"]
