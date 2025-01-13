import yaml
import numpy as np
from astropy.io import fits
from pathlib import Path

class Reader:

    def __init__(self, filepath: str):

        self.filepath = filepath
        self.data = self._load_yaml()

    def _load_yaml(self) -> dict:

        try:
            with open(self.filepath, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.filepath} not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error while parsing input file: {e}")

    def get_section(self, section: str, index: int = None) -> dict:

        keys = section.split(".")
        current_level = self.data
        for key in keys:
            if key not in current_level:
                raise KeyError(f"Section '{section}' not found in input file.")
            current_level = current_level[key]

        if index is not None:
            if not isinstance(current_level, dict):
                raise TypeError(f"Section '{section}' must be a dictionary to support indexing.")

            # Sostituisci le liste o array con i valori indicizzati
            processed_section = {}
            for sub_key, value in current_level.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    try:
                        processed_section[sub_key] = value[index]
                    except IndexError:
                        raise IndexError(f"Index {index} is out of bounds for key '{sub_key}' in section '{section}'.")
                else:
                    raise TypeError(f"Value associated with key '{sub_key}' is not a list or array.")
            return processed_section

        return current_level

    def get_full_config(self) -> dict:

        return self.data


class ParamsReader(Reader):

    pass


class ObsSpecReader(Reader):

    pass


class DataReader(Reader):

    def __init__(self, filepath: str):

        super().__init__(filepath)

        root_dir = Path(__file__).resolve().parents[1]
        self.dat_dir_main = Path(root_dir, Path('data'),
                                 Path(self.data['sample']))

        self.data_dict = {'GCspectro': None}
        self.data_spectro_fiducial_cosmo = {}

    def read_GCspectro(self):

        root = self.data['GCspectro']['root']
        redshifts = self.data['GCspectro']['redshifts']

        if 'cov_is_num' not in self.data['GCspectro'].keys():
            self.data['GCspectro']['cov_is_num'] = False

        if self.data['GCspectro']['cov_is_num']:
            if 'cov_nsim' not in self.data['GCspectro'].keys():
                raise Exception('The parameter cov_nsim for spectro data '
                                'must be set when cov_is_num = True')
            if not isinstance(self.data['GCspectro']['cov_nsim'], int):
                raise TypeError('The parameter cov_nsim for spectro data must '
                                'be set to an integer number when '
                                'cov_is_num = True')
            if self.data['GCspectro']['cov_nsim'] <= 0:
                raise ValueError('The parameter cov_nsim for spectro data '
                                 'must be strictly positive')

        if 'z{:s}' not in root:
            raise ValueError('GCspectro file names should contain z{:s} '
                             'string to enable iteration over bins.')

        cur_fname = root.format(redshifts[0])
        full_path = Path(self.dat_dir_main, 'GCspectro', cur_fname)
        fid_cosmo_file = fits.open(full_path)
        try:
            omnuh2 = 0.0006451438915397982
            self.data_spectro_fiducial_cosmo = {
                'H0': fid_cosmo_file[1].header['HUBBLE'] * 100.0,
                'omch2': ((fid_cosmo_file[1].header['OMEGA_M'] -
                           fid_cosmo_file[1].header['OMEGA_B']) *
                          fid_cosmo_file[1].header['HUBBLE']**2 - omnuh2),
                'ombh2': (fid_cosmo_file[1].header['OMEGA_B'] *
                          fid_cosmo_file[1].header['HUBBLE']**2),
                'ns': fid_cosmo_file[1].header['INDEX_N'],
                'sigma8': fid_cosmo_file[1].header['SIGMA_8'],
                'w': fid_cosmo_file[1].header['W_STATE'],
                'omkh2': (fid_cosmo_file[1].header['OMEGA_K'] *
                          fid_cosmo_file[1].header['HUBBLE']**2),
                # OU-LE3 spectro files always with omnuh2 = 0
                'omnuh2': omnuh2,
                'Omnu': omnuh2 / fid_cosmo_file[1].header['HUBBLE']**2}
            # Omega_radiation is ignored here
            fid_cosmo_file.close()
        except ReaderError:
            log_critical('There was an error when reading the fiducial '
                         'data from OU-level3 files in read_GCspectro')

        GCspectro_dict = {}

        if self.data['GCspectro']['Fourier']:
            k_fac = self.data_spectro_fiducial_cosmo['H0'] / 100.0
            p_fac = 1.0 / (k_fac ** 3.0)
            cov_fac = p_fac ** 2.0

            for z_label in redshifts:
                cur_it_fname = root.format(z_label)
                cur_full_path = Path(self.dat_dir_main, 'GCspectro',
                                     cur_it_fname)
                fits_file = fits.open(cur_full_path)
                average = fits_file[1].data
                kk = average['SCALE_1DIM'] * k_fac
                pk0 = average['AVERAGE0'] * p_fac
                pk2 = average['AVERAGE2'] * p_fac
                pk4 = average['AVERAGE4'] * p_fac

                cov = fits_file[2].data['COVARIANCE'] * cov_fac
                cov_k_i = fits_file[2].data['SCALE_1DIM-I'] * k_fac
                cov_k_j = fits_file[2].data['SCALE_1DIM-J'] * k_fac
                cov_l_i = fits_file[2].data['MULTIPOLE-I']
                cov_l_j = fits_file[2].data['MULTIPOLE-J']

                nk = len(kk)
                cov = np.reshape(cov, newshape=(3 * nk, 3 * nk))
                cov_k_i = np.reshape(cov_k_i, newshape=(3 * nk, 3 * nk))
                cov_k_j = np.reshape(cov_k_j, newshape=(3 * nk, 3 * nk))
                cov_l_i = np.reshape(cov_l_i, newshape=(3 * nk, 3 * nk))
                cov_l_j = np.reshape(cov_l_j, newshape=(3 * nk, 3 * nk))

                GCspectro_dict['{:s}'.format(z_label)] = {'k': kk,
                                                          'pk0': pk0,
                                                          'pk2': pk2,
                                                          'pk4': pk4,
                                                          'cov': cov,
                                                          'cov_k_i': cov_k_i,
                                                          'cov_k_j': cov_k_j,
                                                          'cov_l_i': cov_l_i,
                                                          'cov_l_j': cov_l_j}

                fits_file.close()

        self.data_dict['GCspectro'] = GCspectro_dict
        return
