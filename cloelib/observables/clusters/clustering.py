class HaloClustering:
    def __init__(
        self,
        pertrurbations: Perturbations,
    ):
        self.cosmo = _tempPerturbationsCluster(pertrurbations)

    def WF_ra(self, z_array, r_array, k_array):
        r"""
        Computes the window function and the volume of the spherical shells as a function of the radial separation

        Parameters
        ----------
        z_array: float or numpy.ndarray
                 Redshift at which apply the geometrical correction (Alcock-Paczynski effect)
        k_array: float or numpy.ndarray
                 Wavenumber used to evaluate power spectrum
                 Units: h Mpc^{-1}
        r_array: float or numpy.ndarray
                 Radial separation bins
        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i is the redshift bin, j is the radial bin and k are the wavenumbers
        spherical shell volume: numpy.ndarray
                V[i,j] where i is the redshift bin and j is the radial bin
        """
        z_array = z_array[:, np.newaxis, np.newaxis]
        r_array = r_array[np.newaxis, :, np.newaxis]
        k_array = k_array[np.newaxis, np.newaxis, :]

        r_z_array = (
            self.Dv_rs_func(z_array) * r_array
        )  # AP correction (adds a redshift dependence)
        r3_TH_filter = (
            r_z_array**3
            * 3.0
            * (
                np.sin(k_array * r_z_array)
                - k_array * r_z_array * np.cos(k_array * r_z_array)
            )
            / (k_array * r_z_array) ** 3.0
        )

        W_rad = (r3_TH_filter[:, 1:, :] - r3_TH_filter[:, :-1, :]) / (
            r_z_array[:, 1:, :] ** 3 - r_z_array[:, :-1, :] ** 3
        )

        V_rad = (
            4.0
            * np.pi
            / 3.0
            * ((r_z_array[:, 1:, 0]) ** 3 - (r_z_array[:, :-1, 0]) ** 3)
        )

        return W_rad, V_rad

    # cosmo correction
    def Dv_rs_func(self, z):
        """
        Compute the correction that accounts for the wrong cosmology assumed in the measurement of the 2ptCF
        See https://arxiv.org/pdf/1511.00012.pdf (Sect. 4.3.1) for details.

        Parameters
        ----------
        z: redshift

        Returns
        -------
        DV_over_rs: volume distance over drag scale (sound horizon scale at recombination)

        """

        # here we can work in Mpc (without h unit conversion), as the output is dimensionless

        # isotropic volume distance
        Dv = (
            (1 + z) ** 2
            * self.cosmo["d_z_func"](z) ** 2
            * self.cosmo["c"]
            * z
            / self.cosmo["H_z_func"](z)
        ) ** (1 / 3.0)

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = (
            (1 + z) ** 2
            * self.cosmo["fid_d_z_func"](z) ** 2
            * self.cosmo["c"]
            * z
            / self.cosmo["fid_H_z_func"](z)
        ) ** (1 / 3.0)

        return (Dv / self.cosmo["rdrag"]) * (self.cosmo["fid_rdrag"] / Dv_fid)
