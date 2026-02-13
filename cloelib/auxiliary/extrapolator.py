"""Extrapolator module."""

import numpy as np


def extend_spectra(
    wavenumber_in,
    redshift_in,
    boost_in,
    flag_range,
    norm_dist=0,
    option_wavenumber_low="hm_simple",
    option_wavenumber_high="hm_simple",
    option_redshift="hm_simple",
    option_cosmo="hm_simple",
    extrap_func=None,
    extrap_kmax=500.0,
    extrap_kmin=1e-4,
    extrap_z=4.0,
    wavenumber_tanh_slope=10.0,
    wavenumber_tanh_scale=1.15,
    ns=0.96,
):
    """Calculate extrapolation of the spectrum/boost outside its given range.

    Return spectrum/boost array and corresponding scales in 1/Mpc.

    Options for wavenumber extrapolation:
        - const, `power_law`, `hm_simple`, `hm_smooth`
    Options for redshift extrapolation:
        - const, `power_law`, `hm_simple`
    Options for cosmo extrapolation:
        - const, `hm_simple`, `hm_smooth`

    Both hm_simple and hm_smooth extrapolate with HMcode, but the smooth
    case uses a tanh to interpolate between another extrapolation and
    HMcode, making it a continuous extrapolation. Any of this cases requires a
    function computing the HMcode prediction to be passed

    Parameters
    ----------
    wavenumber_in: numpy.ndarray
        Scales used by emulator in units of 1/Mpc
    redshift_in: float
        Max redshift of emulator
    boost_in: numpy.ndarray
       Array with boost at `z_win` redshifts and scales `wavenumber_in`
    flag_range: bool
        Flag for cosmo params range of emulator
    option_wavenumber_low: string
        Option for low wavenumber extrapolation
    option_wavenumber_high: string
        Option for high wavenumber extrapolation
    option_redshift: str
        Option for redshift extrapolation
    option_cosmo: str
        Option for cosmology extrapolation
    extrap_func: callable
        function to call to get prediciton from HMcode (ignored if not using
        HMcode extrapolation)
    extrap_kmax: float
        max wavenumber for extrapolation in 1/Mpc
    extrap_kmin: float
        min wavenumber for extrapolation in 1/Mpc
    extrap_z: float or np.ndarray
        max redshift to extrapolate or array with redshifts to evaluate

    Returns
    -------
    wavenumber_out: numpy.ndarray
       Concatenation of scales used by emulator with extended ones, in units
       of 1/Mpc
    redshift_out: numpy.ndarray
       Concatenation of redshifts used by emulator with extended ones
    boost_out: numpy.ndarray
       Array with extrapolated boost at extended redshifts
       and scales `wavenumber_out`

    """
    default_n_k = 500
    wavenumber_base = np.geomspace(extrap_kmin, extrap_kmax, default_n_k)

    if isinstance(extrap_z, (int, float)):
        default_n_z = 100
        redshift_base = np.linspace(0, extrap_z, default_n_z)
    elif isinstance(extrap_z, (np.ndarray, list)):
        redshift_base = np.asarray(extrap_z)

    # Extrapolate everything with HMcode if cosmology is not in range
    # and the option to extrapolate is HMcode.
    if (not flag_range) and option_cosmo == "hm_simple":
        # Temporary thing until I have an example case
        wavenumber_out, redshift_out, boost_out = extrap_func()

    # If cosmology is in range, then proceed to do the other extrapolations
    # Or if the cosmology extrapolation option is const or hm_smooth, then we
    # still need to perform extrapolation in the other variables, so do the
    # same thing.
    elif flag_range or (option_cosmo in ["const", "hm_smooth"]):
        # Split the wavenumber range into 3 parts, one is just
        # wavenumber_in, others are:
        # The part with wavenumber<wavenumber_in
        wavenumber_minus = wavenumber_base[wavenumber_base < wavenumber_in[0]]

        # The part with wavenumber>wavenumber_in
        wavenumber_plus = wavenumber_base[wavenumber_base > wavenumber_in[-1]]

        # Join them together for output
        wavenumber_out = np.concatenate(
            (wavenumber_minus, wavenumber_in, wavenumber_plus)
        )

        # For redshift, I assume zmin is always z=0 and is included in emulator
        redshift_plus = redshift_base[redshift_base > redshift_in[-1]]

        # Join them together for output
        redshift_out = np.concatenate((redshift_in, redshift_plus))
        boost_out = np.ones((len(redshift_out), len(wavenumber_out)))

        # Use correct result in the range of the emulator in redshift and
        # wavenumber
        boost_out[
            : len(redshift_in),
            len(wavenumber_minus) : (len(wavenumber_minus) + len(wavenumber_in)),
        ] = boost_in

        if len(redshift_plus) > 0:
            if option_redshift == "hm_simple":
                # Use HMCode for those
                boost_out[
                    len(redshift_in) :,
                    len(wavenumber_minus) : (
                        len(wavenumber_minus) + len(wavenumber_in)
                    ),
                ] = extrap_func(redshift_in=redshift_plus, wavenumber_in=wavenumber_in)[
                    2
                ]

            elif option_redshift == "const":
                # Use const
                boost_out[
                    len(redshift_in) :,
                    len(wavenumber_minus) : (
                        len(wavenumber_minus) + len(wavenumber_in)
                    ),
                ] = (
                    np.ones((len(redshift_plus), len(wavenumber_in)))
                    * boost_in[-1, :][None, :]
                )

            elif option_redshift == "power_law":
                # Use power law

                n_extra_b = (np.log(boost_in[-1, :]) - np.log(boost_in[-2, :])) / (
                    np.log(1 + redshift_in[-1]) - np.log(1 + redshift_in[-2])
                )

                boost_out[
                    boost_in.shape[0] :,
                    len(wavenumber_minus) : (
                        len(wavenumber_minus) + len(wavenumber_in)
                    ),
                ] = (
                    boost_in[-1, :][None, :]
                    * (((1 + redshift_plus) / (1 + redshift_in[-1]))[:, None])
                    ** n_extra_b[None, :]
                )

            else:
                raise Exception("Wrong redshift extrapolation option.")

        # Assume nothing needed for low k region so keep result = 1
        # May need to consider changing that in the future

        # Different options for wavenumber>wavenumber_in
        if wavenumber_base[-1] > wavenumber_in[-1]:
            if option_wavenumber_high == "const":
                # Use final boost for wavenumber>wavenumber_in
                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)) :] = (
                    np.ones((len(redshift_out), len(wavenumber_plus)))
                    * boost_out[:, (len(wavenumber_minus) + len(wavenumber_in) - 1)][
                        :, None
                    ]
                )

            elif option_wavenumber_high == "hm_simple":
                # Use HMCode for wavenumber>wavenumber_in
                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)) :] = (
                    extrap_func(
                        redshift_in=redshift_out, wavenumber_in=wavenumber_plus
                    )[2]
                )

            elif option_wavenumber_high == "hm_smooth":
                # Use modulated HMCode for wavenumber>wavenumber_in
                boost_hmcode = extrap_func(
                    redshift_in=redshift_out, wavenumber_in=wavenumber_plus
                )[2]

                i_last = len(wavenumber_minus) + len(wavenumber_in)

                # Compute boost spectral index at last point
                n_extra_b = (
                    np.log(boost_out[:, i_last - 1]) - np.log(boost_out[:, i_last - 2])
                ) / (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                # Get a power-law extrapolation
                boost_powerlaw = (
                    boost_out[:, i_last - 1][:, None]
                    * ((wavenumber_plus / wavenumber_in[-1])[None, :])
                    ** n_extra_b[:, None]
                )

                # Mix power-law with HMcode using tanh
                # tanh params
                tanh_slope = wavenumber_tanh_slope
                tanh_scale = wavenumber_tanh_scale * np.log(wavenumber_in[-1])

                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)) :] = (
                    boost_powerlaw
                    + 0.5
                    * (boost_hmcode - boost_powerlaw)
                    * (
                        np.tanh(tanh_slope * (np.log(wavenumber_plus) - tanh_scale))
                        + 1.0
                    )[None, :]
                )

            elif option_wavenumber_high == "power_law":
                # Use power law in wavenumber for wavenumber>wavenumber_in

                i_last = len(wavenumber_minus) + len(wavenumber_in)

                n_extra_b = (
                    np.log(boost_out[:, i_last - 1]) - np.log(boost_out[:, i_last - 2])
                ) / (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)) :] = (
                    boost_out[:, i_last - 1][:, None]
                    * ((wavenumber_plus / wavenumber_in[-1])[None, :])
                    ** n_extra_b[:, None]
                )

            elif option_wavenumber_high == "logk2":
                # Use logk2 in wavenumber for wavenumber>wavenumber_in

                i_last = len(wavenumber_minus) + len(wavenumber_in)

                k3Pk = wavenumber_in[-1] ** (4 - ns) * boost_out[:, i_last - 1]

                dP_dlogk = (
                    wavenumber_in[-1] ** (4 - ns) * boost_out[:, i_last - 1]
                    - wavenumber_in[-2] ** (4 - ns) * boost_out[:, i_last - 2]
                ) / (np.log(wavenumber_in[-1]) - np.log(wavenumber_in[-2]))

                Amp_logk2 = 0.25 * dP_dlogk**2 / k3Pk

                logkstar_logk2 = np.log(wavenumber_in[-1]) - 2 * k3Pk / dP_dlogk

                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)) :] = (
                    wavenumber_plus[None, :] ** -(4 - ns)
                    * Amp_logk2[:, None]
                    * (np.log(wavenumber_plus[None, :]) - logkstar_logk2[:, None]) ** 2
                )

            elif option_wavenumber_high == "linear":
                # Linear extrapolation in log-log space (high-k)
                nfit = 5
                logk_plus_fit = np.log(wavenumber_in[-nfit:])
                logP_plus_fit = np.log(boost_in[:, -nfit:]) 
                p = np.polyfit(logk_plus_fit, logP_plus_fit.T, 1)
                
                logk_plus = np.log(wavenumber_plus)
                logP_plus = p[0, :, None] * logk_plus[None, :] + p[1, :, None]
                
                boost_out[:, (len(wavenumber_minus) + len(wavenumber_in)):] = np.exp(logP_plus)

            else:
                raise Exception("Wrong wavenumber extrapolation option.")

        if (wavenumber_base[0] < wavenumber_in[0]) & (option_wavenumber_low == "logk2"):
            # Use power law for wavenumber<wavenumber_in
            i_first = len(wavenumber_minus)

            n_extra_b = (
                np.log(boost_out[:, i_first + 1]) - np.log(boost_out[:, i_first])
            ) / (np.log(wavenumber_in[1]) - np.log(wavenumber_in[0]))

            boost_out[:, : len(wavenumber_minus)] = (
                boost_out[:, i_first][:, None]
                * ((wavenumber_minus / wavenumber_in[0])[None, :]) ** n_extra_b[:, None]
            )

        elif (wavenumber_base[0] < wavenumber_in[0]) & (option_wavenumber_low == "linear"):
            # Linear extrapolation in log-log space (low-k)

            nfit = 5
            logk_minus_fit = np.log(wavenumber_in[:nfit])
            logP_minus_fit = np.log(boost_in[:, :nfit]) 
            p = np.polyfit(logk_minus_fit, logP_minus_fit.T, 1)
            
            logk_minus = np.log(wavenumber_minus)
            logP_minus = p[0, :, None] * logk_minus[None, :] + p[1, :, None]
            boost_out[:, :len(wavenumber_minus)] = np.exp(logP_minus)

        if (not flag_range) and option_cosmo == "hm_smooth":
            boost_hmcode = extrap_func(
                redshift_in=redshift_out, wavenumber_in=wavenumber_out
            )[1]

            dist_trans = np.min(2 * norm_dist, initial=1)

            boost_out = boost_out * (1 - dist_trans) + boost_hmcode * dist_trans

    else:
        raise Exception("Wrong cosmo extrapolation option.")

    return wavenumber_out, redshift_out, boost_out
