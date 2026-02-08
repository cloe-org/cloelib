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
        r"""Class defining the selection function of galaxy clusters, including
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
            Used for computation of Prob(lambda_obs, z_obs)*completeness/purity.
        """
        self.lambda_true_distribution = lambda_true_distribution
        self._sel_cl_data = sel_cl_data
        self._prob_contains_completeness = prob_contains_completeness
        self._extrapolate = extrapolate

    def _compute_prob_comp_pur(self, z_obs_edges, lambda_obs_edges):
        """Computes Prob(lambda_obs, z_obs)*completeness/purity
        with SEL_CL data shaped to contain obs bins ranges.

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.

        Returns
        -------
        prob_data: dict
            SEL_CL data reshaped with obs bins

                * z_obs: expanded observed redshift values
                * lambda_obs: expanded observed richenss values
                * prob_comp_pur: prob_lambda_z_obs*completeness/purity
                * z_obs_bins_slices: slices for each z obs bins in prob_comp_pur
                * lambda_obs_bins_slices: slices for each lambda obs bins in prob_comp_pur

        Note
        ----
            It assumes all tiles have the same ranges and binning!
        """

        # output instanciated with quantities that remain the same:
        prob_data = {}

        #################
        # Reshaped arrays
        #################

        prob_data["z_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["z_obs"],
            self._sel_cl_data["step_size"]["z_obs"],
            z_obs_edges[0],
            z_obs_edges[-1],
        )
        prob_data["lambda_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["lambda_obs"],
            self._sel_cl_data["step_size"]["lambda_obs"],
            lambda_obs_edges[0],
            lambda_obs_edges[-1],
        )

        ######################################################
        # Computes Prob(lambda_obs, z_obs)*completeness/purity
        ######################################################

        # finds which slices correspond to the original arrays
        # i. e. array == expanded_array[slice]
        _zobs_orig_slice = self._get_bin_slices(
            prob_data["z_obs"],
            self._sel_cl_data["arrays"]["z_obs"][[0, -1]],
            endpoint=True,
        )[0]
        _lobs_orig_slice = self._get_bin_slices(
            prob_data["lambda_obs"],
            self._sel_cl_data["arrays"]["lambda_obs"][[0, -1]],
            endpoint=True,
        )[0]
        # make tuple to fill output with data from self._sel_cl_data
        _fill = (
            slice(self._sel_cl_data["area_tile"].size),
            _zobs_orig_slice,
            _lobs_orig_slice,
            slice(self._sel_cl_data["arrays"]["z_true"].size),
            slice(self._sel_cl_data["arrays"]["lambda_true"].size),
        )

        # Istanciate output
        prob_data["prob_comp_pur"] = self._extrapolate * np.ones(
            (
                self._sel_cl_data["area_tile"].size,
                prob_data["z_obs"].size,
                prob_data["lambda_obs"].size,
                self._sel_cl_data["arrays"]["z_true"].size,
                self._sel_cl_data["arrays"]["lambda_true"].size,
            )
        )

        # Add prob_lambda_z_obs
        prob_data["prob_comp_pur"][_fill] = self._sel_cl_data["tables"][
            "prob_lambda_z_obs"
        ]

        # Divide by purity, seting 0 to NaN to avoid dividing by 0
        _pur_reshaped = self._sel_cl_data["tables"]["purity"][:, :, :, None, None]
        _pur_reshaped = np.where(_pur_reshaped == 0, np.nan, _pur_reshaped)
        prob_data["prob_comp_pur"][_fill] /= _pur_reshaped

        ## Do we want to put the division to 0? If YES:
        ## prob_data["prob_comp_pur"][_fill] = (
        ##   np.where(pur_reshaped != 0, prob_data["prob_comp_pur"][_fill], 0.0)
        ## )

        # Add completeness?
        if not self._prob_contains_completeness:
            prob_data["prob_comp_pur"] *= self._sel_cl_data["tables"]["completeness"][
                :, np.newaxis, np.newaxis, :, :
            ]

        ##################
        # For integrations
        ##################

        # Find slices that return the correct range for each obs bins
        prob_data["lambda_obs_bins_slices"] = self._get_bin_slices(
            prob_data["lambda_obs"], lambda_obs_edges
        )
        prob_data["z_obs_bins_slices"] = self._get_bin_slices(
            prob_data["z_obs"], z_obs_edges
        )

        return prob_data

    def _build_windows_interpolators(self, z_obs_edges, lambda_obs_edges):
        r"""Computes the integral over z_obs_edges and lambda_obs_edges of
        Prob(lambda_obs, z_obs)*completeness/purity*Omega_alpha/Omega_tot.
        Builds the interpolators over (z_true, lambda_true) for all bins in
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
        interpolators: list[list[RectBivariateSpline]]
            Interpolator of W(z_true, lambda_true) per (z_obs_bins, lambda_obs_bins).
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

        # Prob*completeness/purity formatted with obs bins
        prob_data = self._compute_prob_comp_pur(z_obs_edges, lambda_obs_edges)

        # Integrate with Omega_alpha/Sum(Omega_alpha)

        integrand = (
            np.expand_dims(self._sel_cl_data["area_tile"], axis=(1, 2, 3, 4))
            * prob_data["prob_comp_pur"]
        ).sum(axis=0) / self._sel_cl_data["area_tile"].sum()

        window_ltrue = np.zeros(
            (
                len(z_obs_edges) - 1,
                len(lambda_obs_edges) - 1,
                len(self._sel_cl_data["arrays"]["z_true"]),
                len(self._sel_cl_data["arrays"]["lambda_true"]),
            )
        )
        for ztab, z_slice in enumerate(prob_data["z_obs_bins_slices"]):
            for ltab, l_slice in enumerate(prob_data["lambda_obs_bins_slices"]):
                window_ltrue[ztab, ltab, :, :] = integrate.simpson(
                    integrate.simpson(
                        integrand[z_slice, l_slice, :, :],
                        x=prob_data["z_obs"][z_slice],
                        axis=0,
                    ),
                    x=prob_data["lambda_obs"][l_slice],
                    axis=0,
                )

        # build interpolator

        interpolators = [
            [
                interpolate.RectBivariateSpline(
                    self._sel_cl_data["arrays"]["z_true"],
                    self._sel_cl_data["arrays"]["lambda_true"],
                    window_ltrue_zobs_lobs,
                )
                for window_ltrue_zobs_lobs in window_ltrue_zobs
            ]
            for window_ltrue_zobs in window_ltrue
        ]

        return interpolators

    def window_redshift_richness_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, mass, lambda_true
    ):
        r"""Computes the window function for observed redshift and richness bins, i. e.:

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
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        """
        ################################################################
        # Compute W_{Delta lambda_obs, Delta z_obs}(lambda_true, z_true)
        ################################################################

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

        ######################################################
        # Compute W_{Delta lambda_obs, Delta z_obs}(M, z_true)
        ######################################################

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
        """Expands lower/upper boundaries of array.

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
        """Finds slices that return the correct range for each bin.

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
        extrapolate=0,
    )
    interps = sfi._build_windows_interpolators(
        lambda_obs_edges=np.array([20.0, 30.0, 45.0, 60.0, 220.0]),
        z_obs_edges=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),
    )
    print(interps[1][1]([10, 20, 30, 40], [0.3, 0.31]))
