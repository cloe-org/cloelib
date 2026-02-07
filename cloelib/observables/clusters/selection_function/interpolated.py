# General imports
import jax.numpy as np  # type: ignore
import numpy as np  # type: ignore
from scipy import integrate, interpolate
from scipy.integrate import simps

from cloelib.observables.clusters.selection_function.lambda_true_distribution import (
    LambdaTrueDistribution,
)


class InterpolatedSelectionFunction:

    def __init__(
        self,
        lambda_true_distribution: LambdaTrueDistribution,
        sel_cl_data=None,
        prob_contains_completeness=True,
        extrapolate=None,
    ):
        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        Parameters
        ----------
        lambda_true_distribution: LambdaTrueDistribution,
            Object that contains the distribution of true richness given mass
        sel_cl_data: dict
            Object that read the SEL_CL output file and formats its accordingly. It must contain the keys:

                * area_tile: area of each homogeneous region
                * arrays: arrays of tabulation
                    * z_obs: observed redshift values
                    * lambda_obs: observed richenss values
                    * z_true: true redshift values
                    * lambda_true: true richenss values
                * tables:
                    * prob_lambda_z_obs: P(lambda_obs, z_obs|lambda_true, z_true)
                    * completeness: completeness(lambda_true, z_true)
                    * purity: purity(ambda_obs, z_obs)
                * step_size:
                    * z_obs: size of steps in z_obs array
                    * lambda_obs: size of steps in lambda_obs array

        prob_contains_completeness : bool
            If sel_cl_data["prob_lambda_z_obs"] already accounts for the completeness.
        extrapolate : float, None
            Behaviour for when z/lambda obs bins are outside the values contained in sel_cl_data.
            If float, sets the float value when out of bounds, if None raises an error.
            Used for prob_lambda_z_obs and purity.

        """
        self.lambda_true_distribution = lambda_true_distribution
        self._sel_cl_data = sel_cl_data
        self.prob_contains_completeness = prob_contains_completeness
        self._extrapolate = extrapolate

    def _reshape_data_with_obs_bins(self, z_obs_edges, lambda_obs_edges):
        """Reshapes SEL_CL data with obs bins and computes
        Pα(λobs, zobs|λtr,ztr)/pα(λobs,zobs)*cα(λtr,ztr)

        Deals with min and max in lobs and zobs:

            * max(lobs_file) might be < max(lambda_obs_edges)
            * min(lobs_file) might be > min(lambda_obs_edges)
            * max(zobs_file) might be < max(z_obs_edges)
            * min(zobs_file) might be > min(z_obs_edges)

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.

        Returns
        -------
        out_data: dict
            SEL_CL data reshaped with obs bins

                * area_tile: area of each homogeneous region
                * arrays: arrays of tabulation
                    * z_obs: expanded observed redshift values
                    * lambda_obs: expanded observed richenss values
                    * z_true: true redshift values
                    * lambda_true: true richenss values
                * tables: new tables expanded with lambda_obs_edges, z_obs_edges
                    * prob_lambda_z_obs: P(lambda_obs, z_obs|lambda_true, z_true)
                    * completeness: completeness(lambda_true, z_true)
                    * purity: purity(ambda_obs, z_obs)
                    * prob_comp_pur: prob_lambda_z_obs*completeness/purity
                * obs_bins_slices: slices that return the correct range for each obs bins
                    * z: slices for z_obs_edges
                    * lambda: slices for lambda_obs_edges

        Note
        ----
            It assumes all tiles have the same ranges and binning!
        """

        # output instanciated with quantities that remain the same:
        out_data = {
            "area_tile": self._sel_cl_data["area_tile"],
            "arrays": {
                key: self._sel_cl_data["arrays"][key]
                for key in ("z_true", "lambda_true")
            },
            "tables": {"completeness": self._sel_cl_data["tables"]["completeness"]},
        }

        #################
        # Reshaped arrays
        #################

        out_data["arrays"]["z_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["z_obs"],
            self._sel_cl_data["step_size"]["z_obs"],
            z_obs_edges[0],
            z_obs_edges[-1],
        )
        out_data["arrays"]["lambda_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["lambda_obs"],
            self._sel_cl_data["step_size"]["lambda_obs"],
            lambda_obs_edges[0],
            lambda_obs_edges[-1],
        )

        #################
        # Reshaped tables
        #################

        # finds which slices correspond to the original arrays
        # i. e. array == expanded_array[slice]
        _lobs_orig_slice = self._get_bin_slices(
            out_data["arrays"]["lambda_obs"],
            self._sel_cl_data["arrays"]["lambda_obs"][[0, -1]],
            endpoint=True,
        )[0]
        _zobs_orig_slice = self._get_bin_slices(
            out_data["arrays"]["z_obs"],
            self._sel_cl_data["arrays"]["z_obs"][[0, -1]],
            endpoint=True,
        )[0]

        # Re-arrange ranges for prob_lambda_z_obs
        out_data["tables"]["prob_lambda_z_obs"] = self._extrapolate * np.ones(
            (
                out_data["area_tile"].size,
                out_data["arrays"]["z_obs"].size,
                out_data["arrays"]["lambda_obs"].size,
                out_data["arrays"]["z_true"].size,
                out_data["arrays"]["lambda_true"].size,
            )
        )
        out_data["tables"]["prob_lambda_z_obs"][
            :, _zobs_orig_slice, _lobs_orig_slice, :, :
        ] = self._sel_cl_data["tables"]["prob_lambda_z_obs"]

        # Re-arrange ranges for purity
        out_data["tables"]["purity"] = self._extrapolate * np.ones(
            (
                out_data["area_tile"].size,
                out_data["arrays"]["z_obs"].size,
                out_data["arrays"]["lambda_obs"].size,
            )
        )
        out_data["tables"]["purity"][:, _zobs_orig_slice, _lobs_orig_slice] = (
            self._sel_cl_data["tables"]["purity"]
        )

        #####################################################################
        # Compute Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)
        #####################################################################

        # Evaluate multiplication for each tile
        # Pα(λobs|λtr,ztr) * Pα(zobs|λtr,ztr) / Pα(λobs,zobs) * Cα(λtr,ztr)
        # Dimensions: (lob, ztr, ltr)*(zob, ztr, ltr)*(zob, lob)*(ztr, ltr) --> (zob,lob,ztr,ltr)

        ## To avoid dividing by 0: putting elements with 0 values to NaN
        _pur_reshaped = out_data["tables"]["purity"][:, :, :, None, None]
        _pur_reshaped = np.where(_pur_reshaped == 0, np.nan, _pur_reshaped)

        # compute multiplication
        out_data["tables"]["prob_comp_pur"] = (
            out_data["tables"]["prob_lambda_z_obs"] / _pur_reshaped
        )
        if not self.prob_contains_completeness:
            out_data["tables"]["prob_comp_pur"] *= out_data["tables"]["completeness"][
                :, None, None, :, :
            ]

        ## Do we want to put the division to 0? If YES:
        ## out_data["tables"]["prob_comp_pur"] = (
        ##   np.where(pur_reshaped != 0, out_data["tables"]["prob_comp_pur"], 0.0)
        ## )

        ##################
        # For integrations
        ##################

        # Find slices that return the correct range for each obs bins
        out_data["obs_bins_slices"] = {
            "lambda": self._get_bin_slices(
                out_data["arrays"]["lambda_obs"], lambda_obs_edges
            ),
            "z": self._get_bin_slices(out_data["arrays"]["z_obs"], z_obs_edges),
        }

        return out_data

    def _build_windows_interpolators(self, z_obs_edges, lambda_obs_edges):
        r"""
        Selection Function from file.
        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of
        Pα(λobs,zobs|λtr,ztr)/pα(λobs,zobs)*cα(λtr,ztr) Omega_alpha/Omega_tot.
        Builds the interpolators over (ztr,ltr) for all bins in
        z_obs_edges, lambda_obs_edges.


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(\lambda_{\rm true}, z_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}


        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.

        Returns
        -------
        integ4d_interp_func: list[list[RectBivariateSpline]]
            Interpolator of f(z_true, lambda_true) per (z_obs_bins, lambda_obs_bins).
        """

        if self._extrapolate is None:
            err = []
            if z_obs_edges[0] < self._sel_cl_data["arrays"]["z_obs"][0]:
                err.append("lower z_obs_edges")
            if z_obs_edges[-1] > self._sel_cl_data["arrays"]["z_obs"][-1]:
                err.append("upper z_obs_edges")
            if lambda_obs_edges[0] < self._sel_cl_data["arrays"]["lambda_obs"][0]:
                err.append("lower lambda_obs_edges")
            if lambda_obs_edges[-1] > self._sel_cl_data["arrays"]["lambda_obs"][-1]:
                err.append("upper lambda_obs_edges")
            if len(err) > 0:
                err = ",".join(err)
                raise ValueError(f"Cannot use these bins: {err} out of bounds.")

        # Format sel_cl data with obs bins

        sel_cl_data_fmt = self._reshape_data_with_obs_bins(
            z_obs_edges, lambda_obs_edges
        )

        # Integrate sum_a/Omega_tot = tildeI(λtr,ztr,∆λobs,∆zobs)

        integrand = (
            np.expand_dims(sel_cl_data_fmt["area_tile"], axis=(1, 2, 3, 4))
            * sel_cl_data_fmt["tables"]["prob_comp_pur"]
        ).sum(axis=0) / sel_cl_data_fmt["area_tile"].sum()

        integ4d = np.zeros(
            (
                len(z_obs_edges) - 1,
                len(lambda_obs_edges) - 1,
                len(sel_cl_data_fmt["arrays"]["z_true"]),
                len(sel_cl_data_fmt["arrays"]["lambda_true"]),
            )
        )
        for ltab, l_slice in enumerate(sel_cl_data_fmt["obs_bins_slices"]["lambda"]):
            for ztab, z_slice in enumerate(sel_cl_data_fmt["obs_bins_slices"]["z"]):
                integ4d[ztab, ltab, :, :] = integrate.simpson(
                    integrate.simpson(
                        integrand[z_slice, l_slice, :, :],
                        x=sel_cl_data_fmt["arrays"]["z_obs"][z_slice],
                        axis=0,
                    ),
                    x=sel_cl_data_fmt["arrays"]["lambda_obs"][l_slice],
                    axis=0,
                )

        # build interpolator

        integ4d_interp_func = [
            [
                interpolate.RectBivariateSpline(
                    sel_cl_data_fmt["arrays"]["z_true"],
                    sel_cl_data_fmt["arrays"]["lambda_true"],
                    integ4d_zobs_lobs,
                )
                for integ4d_zobs_lobs in integ4d_zobs
            ]
            for integ4d_zobs in integ4d
        ]

        return integ4d_interp_func

    def window_redshift_richness_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, lambda_true
    ):
        r"""
        Computes the window function for observed redshift and richness bins, i. e.:


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        lambda_true : numpy.ndarray
            True richness to compute the window.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        """
        ################################################
        # Compute P(Delta lobs, Delta zobs|ltrue, ztrue)
        ################################################

        interpolators = self._build_windows_interpolators(z_obs_edges, lambda_obs_edges)

        window_lambda_true = np.zeros(
            len(z_obs_edges) - 1,
            len(lambda_obs_edges) - 1,
            z_true.size,
            lambda_true.size,
        )
        for ztab, interp_zobs in enumerate(interpolators):
            for ltab, interp_zobs_lobs in enumerate(interp_zobs):
                window_lambda_true[ztab, ltab] = interp_lobs_zobs(z_true, lambda_true)

        ################################################
        # Compute P(Delta lobs, Delta zobs|mass, ztrue)
        ################################################

        pdf_mass_richness_scaling = self.lambda_true_distribution.prob_richness(
            z_true, mass, lambda_true
        )  # (z, M, lambda_true)

        return simps(
            pdf_mass_richness_scaling[np.newaxis, np.newaxis, :, :, :]
            * window_lambda_true[:, :, :, np.newaxis, :],
            x=lambda_true,
            axis=-1,
        )

    #######
    # Utils
    #######

    @staticmethod
    def _expand_array(array, array_step, lower_value, upper_value):
        """
        Expands lower/upper boundaries of array.

        Parameters
        ----------
        array : np.ndarray
            Original array
        array_step : float
            Step size of original array
        lower_value : float
            Lower value for expansion. Not used if array.min()<lower_value.
        upper_value : float
            Upper value for expansion. Not used if array.max()>upper_value.

        Returns
        -------
        np.ndarray
            Array with expanded boundaries
        """
        upper_addition = np.arange(
            array[-1] + array_step, upper_value + array_step, array_step
        )
        # tick here: make a decreasing array and flip it
        lower_addition = np.arange(
            array[0] - array_step, lower_value - array_step, -array_step
        )[::-1]

        return np.append(lower_addition, np.append(array, upper_addition))

    @staticmethod
    def _get_bin_slices(array, bins_edges, endpoint=False):
        """
        Finds slices that return the correct range for each bin.

        Parameters
        ----------
        array : np.ndarray
            Original array
        bins_edges : list
            Edges of bins.
        endpoint : bool
            Include upper value of edges in slices.

        Returns
        -------
        list[slice]
            List of slices that returns the values in the array
            for each bin of bins_edges.
        """
        # first, find indices at the bin edges
        ind_edges = np.abs(
            array[np.newaxis, :] - np.array(bins_edges)[:, np.newaxis]
        ).argmin(axis=1)

        shift = 1 if endpoint else 0

        return [slice(low, high + shift) for low, high in zip(ind_edges, ind_edges[1:])]


def read_sel_cl_output(sel_cl_filename):
    """Object to read ouput file from SEL_CL.

    Parameters
    ----------
    sel_cl_filename: str
        Name of fits file containing a multi-dimensional of the Selection
        outputted from Sinfonia.

    Returns
    -------
    sel_cl_data: dict
        Object that read the SEL_CL output file and formats its accordingly. It will contain the keys:

                * area_tile: area of each homogeneous region
                * arrays: arrays of tabulation
                    * z_obs: observed redshift values
                    * lambda_obs: observed richenss values
                    * z_true: true redshift values
                    * lambda_true: true richenss values
                * tables:
                    * prob_lambda_z_obs: P(lambda_obs, z_obs|lambda_true, z_true)
                    * completeness: completeness(lambda_true, z_true)
                    * purity: purity(ambda_obs, z_obs)
                * step_size:
                    * z_obs: size of steps in z_obs array
                    * lambda_obs: size of steps in lambda_obs array
    """

    # To be adapted with fitsio - f = fitsio.FITS("your_file.fits")
    from astropy.io import fits

    hdul = fits.open(sel_cl_filename)

    # dictionary to store all outputs
    sel_cl_data = {}

    ############
    # get arrays
    ############

    _arrays_steps = {
        hdul[1].header[f"CTYPE{i}"]: hdul[1].header[f"NAXIS{i}"]
        for i in range(1, 1 + hdul[1].header["NAXIS"])
    }

    sel_cl_data["arrays"] = {
        name.lower(): np.linspace(
            hdul[1].header[f"HIERARCH {name}_START"],
            hdul[1].header[f"HIERARCH {name}_END"],
            num_steps,
            endpoint=True,
        )
        for name, num_steps in _arrays_steps.items()
    }

    ############
    # get tables
    ############

    def get_hdu(extname):
        for hdu in hdul:
            if hdu.header.get("EXTNAME") == extname:
                return hdu
        raise ValueError(f"Missing EXTNAME={extname} hdu from SEL_CL file!")

    # Find number of tiles from fits file
    # this number will eventually be at hdul[0].header
    n_tiles = int((len(hdul) - 1) / 7)

    sel_cl_data["tables"] = {
        name: np.array([get_hdu(f"{extname_pref}{it}").data for it in range(n_tiles)])
        for name, extname_pref in (
            ("prob_lambda_z_obs", "PROB_LAMBDA_Z_OBS_TRUE_"),  # (ltr, ztr, lobs, zobs)
            ("completeness", "COMP_LAMBDA_Z_TRUE_TRUE_"),  # (ltr, ztr)
            ("purity", "PURITY_LAMBDA_Z_OBS_OBS_"),  # (lobs, zobs)
        )
    }

    # make all tables follow the axis order: (zobs, lobs, ztr, ltr)
    sel_cl_data["tables"]["prob_lambda_z_obs"] = sel_cl_data["tables"][
        "prob_lambda_z_obs"
    ].transpose(0, 4, 3, 2, 1)
    sel_cl_data["tables"]["completeness"] = sel_cl_data["tables"][
        "completeness"
    ].transpose(0, 2, 1)
    sel_cl_data["tables"]["purity"] = sel_cl_data["tables"]["purity"].transpose(0, 2, 1)

    ########
    # others
    ########

    # step size of obs quantities
    sel_cl_data["step_size"] = {
        name.lower(): hdul[1].header[f"HIERARCH {name}_STEP"]
        for name in ["Z_OBS", "LAMBDA_OBS"]
    }

    # Tile areas
    sel_cl_data["area_tile"] = np.array(
        [
            get_hdu(f"AREA_{it}").header["HIERARCH EFFECTIVE_AREA"]
            for it in range(n_tiles)
        ]
    )

    return sel_cl_data


if __name__ == "__main__":

    # Read data
    import sys

    from cloelib.observables.clusters.selection_function.lambda_true_distribution import (
        LognormalPowerLawLambdaTrueDistribution,
    )

    print("Test with SEL_CL data")
    if len(sys.argv) == 1:
        raise ValueError("Missing SEL_CL input file")
    in_file = sys.argv[1]

    sfi = InterpolatedSelectionFunction(
        lambda_true_distribution=LognormalPowerLawLambdaTrueDistribution(
            A_l=None,
            B_l=None,
            C_l=None,
            sig_A_l=None,
            sig_B_l=None,
            sig_C_l=None,
        ),
        sel_cl_data=read_sel_cl_output(in_file),
    )
    interps = sfi._build_windows_interpolators(
        lambda_obs_edges=np.array([20.0, 30.0, 45.0, 60.0, 220.0]),
        z_obs_edges=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),
    )
    print(interps[1][1]([10, 20, 30, 40], [0.3, 0.31]))
