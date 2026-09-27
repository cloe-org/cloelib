"""Regenerate the reference-numerics fixture for
`test_photo_tracers_characterization.py`.

Run manually (`python tests/fixtures/generate_photo_tracers_baseline.py`
from the repo root) only when a change is *intended* to alter
`get_window`/`get_Cl` output for these configs. Not collected by pytest.
"""

from pathlib import Path

import numpy as np

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBNonLinearPerturbations
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

H0 = 67.7
h = H0 / 100.0
omch2 = 0.12
Omega_cdm0 = omch2 / h**2
ombh2 = 0.022
Omega_b0 = ombh2 / h**2

background = CAMBBackground(
    H0=H0,
    Omega_b0=Omega_b0,
    Omega_cdm0=Omega_cdm0,
    Omega_k0=0.0,
    As=2e-9,
    ns=0.96,
    alpha_s=0.0,
    mnu=0.06,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.0,
    N_mnu=1,
)

z_auto = np.linspace(0.01, 1100.0, 100)
z = np.linspace(0.2, 2.0, 20)
perturbations = CAMBNonLinearPerturbations(background, None, z_auto)

n_z_bins = 2
dndz = np.ones((n_z_bins, len(z)))
dndz /= np.trapezoid(dndz, z, axis=1)[:, None]

nuisance_shear = {
    **{f"multiplicative_bias_{i + 1}": 0.0 for i in range(n_z_bins)},
    **{f"dz_shear_{i + 1}": 0.0 for i in range(n_z_bins)},
    **{f"width_shear_{i + 1}": 1.0 for i in range(n_z_bins)},
    "AIA": 1.0,
    "CIA": 0.0134,
    "EtaIA": -0.41,
}

nuisance_pos_common = {
    **{f"dz_pos_{i + 1}": 0.0 for i in range(n_z_bins)},
    **{f"width_pos_{i + 1}": 1.0 for i in range(n_z_bins)},
    **{f"magnification_bias_{i + 1}": 0.3 for i in range(n_z_bins)},
}

nuisance_pos_per_bin = {
    **nuisance_pos_common,
    "b1_photo_bin0": 1.1,
    "b1_photo_bin1": 1.4,
}
nuisance_pos_poly = {
    **nuisance_pos_common,
    "b1_photo_poly0": 1.0,
    "b1_photo_poly1": 0.1,
    "b1_photo_poly2": 0.0,
    "b1_photo_poly3": 0.0,
}

shear_tracer = ShearTracer(
    perturbations=perturbations,
    dndz=dndz,
    z=z,
    nuisance_params=nuisance_shear,
    ia_model="NLA",
)
pos_per_bin = PositionsTracer(
    perturbations=perturbations,
    dndz=dndz,
    z=z,
    galaxy_bias_model="per_bin",
    nuisance_params=nuisance_pos_per_bin,
)
pos_per_bin_int = PositionsTracer(
    perturbations=perturbations,
    dndz=dndz,
    z=z,
    galaxy_bias_model="per_bin_int",
    nuisance_params=nuisance_pos_per_bin,
)
pos_poly = PositionsTracer(
    perturbations=perturbations,
    dndz=dndz,
    z=z,
    galaxy_bias_model="poly",
    nuisance_params=nuisance_pos_poly,
)
cmbl_tracer = CMBLensingTracer(perturbations=perturbations, z=z)

out = {}
out["window_shear"] = np.asarray(shear_tracer.get_window(z))
out["window_pos_per_bin"] = np.asarray(pos_per_bin.get_window(z))
out["window_pos_per_bin_int"] = np.asarray(pos_per_bin_int.get_window(z))
out["window_pos_poly"] = np.asarray(pos_poly.get_window(z))
out["window_cmbl"] = np.asarray(cmbl_tracer.get_window(z))

nl = 6
ells = np.logspace(1.0, np.log10(200), nl)
ks = np.asarray(perturbations.k)
out["ells"] = ells
out["ks"] = ks

pairs = {
    "she_she": AngularTwoPoint(shear_tracer, shear_tracer),
    "pos_pos": AngularTwoPoint(pos_per_bin, pos_per_bin),
    "pos_she": AngularTwoPoint(pos_per_bin, shear_tracer),
    "cmbl_cmbl": AngularTwoPoint(cmbl_tracer, cmbl_tracer),
    "cmbl_pos": AngularTwoPoint(cmbl_tracer, pos_per_bin),
    "cmbl_she": AngularTwoPoint(cmbl_tracer, shear_tracer),
}

for prefix, tp in pairs.items():
    cl = tp.get_Cl(ells, 0, ks)
    for key, val in cl.items():
        flat_key = f"cl_{prefix}_" + "_".join(str(k) for k in key)
        out[flat_key] = np.asarray(val.array)

npz_path = Path(__file__).parent / "photo_tracers_baseline.npz"
np.savez(npz_path, **out)
print(f"Wrote {len(out)} arrays to {npz_path}")
for k, v in out.items():
    print(f"  {k}: shape={np.asarray(v).shape}")
