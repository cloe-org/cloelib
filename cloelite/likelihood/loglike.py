import numpy as np

import nautilus

from cloelite.input import reader
from cloelite.masking import masking
from cloelite.cosmology import camb_cosmology
from cloelite.observables import spectro


class EuclidLikelihood:

    def __init__(self, data: str, observable_specifications: str,
                 params_input: str):

        data_reader = reader.DataReader(data)
        data_reader.read_GCspectro()
        data = data_reader.data_dict
        self.k = data['GCspectro']['1.']['k']

        obs_spec_reader = reader.ObsSpecReader(observable_specifications)
        obs_spec = obs_spec_reader.get_full_config()

        self.mask = masking.Masking(data, obs_spec)

        self.data_vec_masked = self.mask.get_masked_data_vector_GCspectro()
        self.cov_mat_masked = self.mask.get_masked_covariance_GCspectro()
        self.inv_cov_mat_masked = np.linalg.inv(self.cov_mat_masked)

        self.params_reader = reader.ParamsReader(params_input)
        H0 = self.params_reader.get_section('cosmology')['H0']
        h = H0 / 100.0
        ombh2 = self.params_reader.get_section('cosmology')['ombh2']
        omch2 = self.params_reader.get_section('cosmology')['omch2']
        ns = self.params_reader.get_section('cosmology')['ns']
        As = self.params_reader.get_section('cosmology')['As']
        background = camb_cosmology.CAMBBackground(H0=H0, ombh2=ombh2,
                                                   omch2=omch2, ns=ns, As=As,
                                                   w=-1.0, wa=0.0, Omk=0.0,
                                                   sigma8=0.0, gamma_MG=0.0)
        linear_perturbations = camb_cosmology.CAMBLinearPerturbations(
            background=background, redshifts=np.linspace(0.0, 4.0, 256))
        background_fiducial = camb_cosmology.CAMBBackground(
            H0=H0, ombh2=ombh2, omch2=omch2, ns=ns, As=As, w=-1.0, wa=0.0,
            Omk=0.0, sigma8=0.0, gamma_MG=0.0)
        self.mps_spectro = spectro.LegendreMultipolesComet(
            linear_perturbations=linear_perturbations,
            background_fiducial=background_fiducial)

    def get_prior(self):
        prior = nautilus.Prior()
        prior.add_parameter('h', dist=(0.6, 0.8))
        prior.add_parameter('omch2', dist=(0.085, 0.15))
        prior.add_parameter('As', dist=(1.5e-9, 2.7e-9))
        return prior

    def loglike(self, params_dict):
        nbar = nbar = np.array([2.042611E-03, 1.02876011E-03, 0.58531983E-03, 0.313402E-03]) * 0.67**3
        theory_vec = []
        for i,z in enumerate([1.0, 1.2, 1.4, 1.65]):
            parameters = self.params_reader.get_section('cosmology') | self.params_reader.get_section('GCsp.theory', i) | self.params_reader.get_section('GCsp.systematics', i)
            parameters['z'] = z
            parameters.update(params_dict)
            self.mps_spectro.set_number_density(nbar=nbar[i])
            mps = self.mps_spectro.power_multipoles(k=self.k, parameters=parameters, ells=[0,2,4])
            for ell in [0,2,4]:
                theory_vec = np.concatenate((theory_vec, mps[f'ell{ell}']))

        theory_vec_masked = self.mask.get_masked_theory_vector_GCspectro(theory_vec)
        diff = theory_vec_masked - self.data_vec_masked

        return -0.5 * np.dot(diff, np.dot(self.inv_cov_mat_masked, diff))
