"""Tests for the T-tilde dataset handling: grids, download and lazy loading.

The real dataset is 1.8 GB, so the tests use tiny fake ones with the same
layout (one directory per matrix, one `T_tilde_l_{ell:.1f}.npy` per
multipole, each of shape `(1, n_chi, n_R, n_cheb)`); the real dataset, when
present, is only used for a few cheap consistency checks.
"""

import hashlib
import tarfile

import jax.numpy as jnp
import numpy as np
import pytest

from cloelib.auxiliary.t_tildes import (
    BLAST_128_64_161,
    DEFAULT_DATA_DIR,
    NonLimberGrid,
    TTildeDataset,
    TTildeStore,
    directory_name,
    ensure_dataset,
)

REAL_DATASET = DEFAULT_DATA_DIR / BLAST_128_64_161.name

N_ELL, N_CHI, N_R, N_CHEB = 3, 4, 5, 6
FAKE_ELLS = np.array([2.0, 14.3, 97.1])


@pytest.fixture
def small_grid():
    """A tiny grid for the fake datasets.

    Its ells (14.2999, 97.0775) are not the file names (14.3, 97.1): the
    store has to match them through the one-decimal rounding.
    """
    return NonLimberGrid(
        ells=jnp.array([2.0, 14.2999, 97.0775]),  # the files carry 1 decimal
        chi=jnp.linspace(26.0, 100.0, N_CHI),
        R=jnp.linspace(0.1, 1.0, N_R),
        k_cheb=jnp.linspace(-4.0, 1.0, N_CHEB),
        w_chi=jnp.ones(N_CHI),
        w_R=jnp.ones(N_R),
    )


def _write_matrix(root, name, seed):
    """One fake T-tilde matrix; returns the array stacked as the store should."""
    rng = np.random.default_rng(seed)
    folder = root / directory_name(name)
    folder.mkdir(parents=True)
    blocks = []
    for ell in FAKE_ELLS:
        block = rng.normal(size=(1, N_CHI, N_R, N_CHEB))
        np.save(folder / f"T_tilde_l_{ell:.1f}.npy", np.asfortranarray(block))
        blocks.append(block[0])
    return np.stack(blocks)


@pytest.fixture
def fake_dataset(tmp_path):
    """Three small fake matrices on disk, with the layout of the real dataset.

    Returns the directory and the arrays the store is expected to return.
    """
    root = tmp_path / "dataset"
    matrices = {
        name: _write_matrix(root, name, seed)
        for seed, name in enumerate(["T_2_00", "T_minus2_00", "T_0_20"])
    }
    return root, matrices


# ---------------------------------------------------------------------------
# directory names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, directory",
    [
        ("T_2_00", "T_tilde_2_00"),
        ("T_0_00", "T_tilde_0_00"),
        ("T_minus2_00", "T_tilde_-2_00"),
        ("T_2_22", "T_tilde_2_22"),
        ("T_0_20", "T_tilde_0_20"),
    ],
)
def test_directory_name(name, directory):
    """A matrix name maps to its directory in the dataset: T_minus2_00 -> T_tilde_-2_00."""
    assert directory_name(name) == directory


@pytest.mark.parametrize("name", ["T_tilde_2_00", "T_2_0", "2_00", "T_-2_00", ""])
def test_directory_name_rejects_malformed_names(name):
    """Only names of the form T_[minus]<power>_<two digits> are accepted."""
    with pytest.raises(ValueError, match="Invalid T-tilde name"):
        directory_name(name)


# ---------------------------------------------------------------------------
# NonLimberGrid
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def blast_grid():
    """The grids of the default dataset (built once per module)."""
    return NonLimberGrid.blast_128_64_161()


def test_blast_grid_sizes(blast_grid):
    """The default dataset has 22 ells, 128 chi, 64 R and 161 Chebyshev coefficients."""
    assert blast_grid.tilde_shape == (22, 128, 64, 161)
    assert blast_grid.w_chi.shape == (128,) and blast_grid.w_R.shape == (64,)


def test_blast_grid_ells(blast_grid):
    """The 22 multipoles are the Chebyshev nodes on [2, 2000] below 220, increasing.

    First and last values are those the T-tilde files were generated for.
    """
    ells = np.asarray(blast_grid.ells)
    np.testing.assert_allclose(ells[:3], [2.0, 2.49294619, 3.9712983], rtol=1e-8)
    np.testing.assert_allclose(ells[-1], 211.63514264, rtol=1e-8)
    assert np.all(np.diff(ells) > 0) and ells.max() < 220.0


def test_blast_grid_chi(blast_grid):
    """chi is uniform on [26, 7000] Mpc with 128 points.

    Uniform spacing is required by the Simpson weights.
    """
    chi = np.asarray(blast_grid.chi)
    assert chi[0] == 26.0 and chi[-1] == 7000.0
    np.testing.assert_allclose(np.diff(chi), (7000.0 - 26.0) / 127, rtol=1e-12)


def test_blast_grid_chi_weights_integrate_chi(blast_grid):
    # w_chi includes the chi measure: sum(w_chi * f) = int chi f dchi
    """w_chi includes the chi measure of the integral: it integrates chi^2 over the grid.

    sum(w_chi * chi) must be (b^3 - a^3) / 3, which tests the Simpson weights
    together with the chi factor and the grid spacing.
    """
    chi = np.asarray(blast_grid.chi)
    a, b = chi[0], chi[-1]
    np.testing.assert_allclose(
        np.sum(np.asarray(blast_grid.w_chi) * chi),  # int chi^2 dchi
        (b**3 - a**3) / 3,
        rtol=1e-6,
    )


def test_blast_grid_R_nodes(blast_grid):
    """R holds the 64 positive nodes of a 129-point Clenshaw-Curtis rule, increasing up to 1.

    These are cos(pi j / 128): the same R the T-tilde were computed at.
    """
    R = np.asarray(blast_grid.R)
    assert np.all(np.diff(R) > 0) and R[-1] == 1.0
    assert R[0] == pytest.approx(0.0245412285229, rel=1e-9)  # cos(pi * 63/128)
    # the 64 positive nodes of the 129-point Clenshaw-Curtis rule on [-1, 1]
    np.testing.assert_allclose(
        R, np.cos(np.pi * (63 - np.arange(64)) / 128), atol=1e-12
    )


def test_blast_grid_R_weights_follow_blast(blast_grid):
    """w_R are the Clenshaw-Curtis weights of the positive nodes, with the smallest one halved.

    The halving follows Blast.jl (a numerical correction for dropping R < 0, not
    the analytic solution); every other weight is the plain CC one.
    """
    from cloelib.auxiliary.chebyshev import clenshaws_curtis_quadrature

    _, w_all = clenshaws_curtis_quadrature(129, -1.0, 1.0)
    w_positive = np.flip(np.asarray(w_all)[:64])  # increasing R
    w_R = np.asarray(blast_grid.w_R)
    # smallest R has its weight halved (Blast.jl), the others are plain CC
    np.testing.assert_allclose(w_R[0], w_positive[0] / 2, rtol=1e-12)
    np.testing.assert_allclose(w_R[1:], w_positive[1:], rtol=1e-12)
    # int_0^1 dR = 1 once the midpoint (R = 0, dropped) is accounted for
    assert 0.9 < w_R.sum() < 1.0


def test_blast_grid_k_nodes(blast_grid):
    """161 Chebyshev nodes in log10 k, between log10(5e-5) and log10(16) (k in 1/Mpc)."""
    k = np.asarray(blast_grid.k_cheb)
    assert len(k) == 161
    np.testing.assert_allclose([k.min(), k.max()], [np.log10(5e-5), np.log10(16.0)])


# ---------------------------------------------------------------------------
# TTildeStore
# ---------------------------------------------------------------------------
def test_store_loads_and_stacks_in_ell_order(fake_dataset, small_grid):
    """The store returns each matrix as (n_ell, n_chi, n_R, n_cheb), ordered as grid.ells.

    The per-ell files (written here in Fortran order, like the real ones) are
    stacked in the order of the grid, whatever order they are listed on disk.
    """
    root, matrices = fake_dataset
    store = TTildeStore(root, grid=small_grid)

    for name, expected in matrices.items():
        loaded = store.get(name)
        assert loaded.shape == small_grid.tilde_shape
        assert loaded.dtype == jnp.float64
        np.testing.assert_array_equal(loaded, expected)


def test_store_matches_files_by_rounded_ell(fake_dataset, small_grid):
    # 14.2999 -> "14.3", 97.0775 -> "97.1": nodes need not equal the file names
    """A grid ell is matched to the file named after its one-decimal rounding.

    14.2999 -> T_tilde_l_14.3.npy and 97.0775 -> T_tilde_l_97.1.npy.
    """
    root, matrices = fake_dataset
    out = TTildeStore(root, grid=small_grid).get("T_2_00")
    np.testing.assert_array_equal(out[1], matrices["T_2_00"][1])
    np.testing.assert_array_equal(out[2], matrices["T_2_00"][2])


def test_store_is_lazy_and_memoized(fake_dataset, small_grid, monkeypatch):
    """Each matrix is about 230 MB: it is read only when asked for, and only once.

    Counts np.load calls: none at construction, one per ell on the first
    get(), none on the second, and the matrices never requested stay untouched.
    """
    root, _ = fake_dataset
    store = TTildeStore(root, grid=small_grid)
    loads = []
    original = np.load
    monkeypatch.setattr(
        np,
        "load",
        lambda path, *a, **k: loads.append(str(path)) or original(path, *a, **k),
    )

    assert loads == []  # nothing read at construction
    first = store.get("T_2_00")
    assert len(loads) == N_ELL and all("T_tilde_2_00" in f for f in loads)

    assert store.get("T_2_00") is first  # memoized: no further reads
    assert len(loads) == N_ELL

    store.get("T_minus2_00")  # only now is the second matrix touched
    assert len(loads) == 2 * N_ELL
    assert not any("T_tilde_0_20" in f for f in loads)


def test_store_missing_directory(fake_dataset, small_grid):
    """Asking for a matrix the dataset does not contain raises FileNotFoundError."""
    root, _ = fake_dataset
    with pytest.raises(FileNotFoundError, match="Missing T-tilde directory"):
        TTildeStore(root, grid=small_grid).get("T_2_22")


def test_store_missing_ell_file(fake_dataset, small_grid):
    """A missing per-ell file is reported with the multipole it was needed for."""
    root, _ = fake_dataset
    (root / "T_tilde_2_00" / "T_tilde_l_14.3.npy").unlink()
    with pytest.raises(FileNotFoundError, match="ell = 14.300"):
        TTildeStore(root, grid=small_grid).get("T_2_00")


def test_store_wrong_shape(fake_dataset, small_grid):
    """A file whose shape does not match the grid is rejected.

    This catches a dataset computed on other grids than the ones in use.
    """
    root, _ = fake_dataset
    np.save(root / "T_tilde_2_00" / "T_tilde_l_2.0.npy", np.zeros((1, N_CHI, N_R, 3)))
    with pytest.raises(ValueError, match="shape"):
        TTildeStore(root, grid=small_grid).get("T_2_00")


def test_store_missing_dataset_path(tmp_path, small_grid):
    """A dataset directory that does not exist is rejected when the store is built."""
    with pytest.raises(FileNotFoundError, match="not found"):
        TTildeStore(tmp_path / "nowhere", grid=small_grid)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_archive(tmp_path):
    """A tiny .tar.gz dataset on disk (served through a file:// URL), and its SHA-256."""
    source = tmp_path / "src" / "T_tildes_tiny"
    _write_matrix(source, "T_2_00", seed=0)
    archive = tmp_path / "T_tildes_tiny.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="T_tildes_tiny")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, sha


def test_ensure_dataset_downloads_verifies_and_extracts(tmp_path, fake_archive):
    """First use: fetch the archive, check its checksum, extract it, remove the archive."""
    archive, sha = fake_archive
    dataset = TTildeDataset("T_tildes_tiny", archive.as_uri(), sha)
    data_dir = tmp_path / "data"

    target = ensure_dataset(dataset, data_dir)

    assert target == data_dir / "T_tildes_tiny"
    assert (target / "T_tilde_2_00" / "T_tilde_l_2.0.npy").is_file()
    assert not (data_dir / "T_tildes_tiny.tar.gz").exists()  # downloaded: cleaned up


def test_ensure_dataset_is_a_no_op_when_present(tmp_path, fake_archive):
    """Once extracted, nothing is downloaded again.

    The dataset is given an unreachable URL and a wrong checksum: the call
    must still succeed, so it never looked at them.
    """
    archive, sha = fake_archive
    dataset = TTildeDataset("T_tildes_tiny", archive.as_uri(), sha)
    data_dir = tmp_path / "data"
    target = ensure_dataset(dataset, data_dir)

    broken = TTildeDataset("T_tildes_tiny", "file:///does/not/exist", "0" * 64)
    assert ensure_dataset(broken, data_dir) == target  # no download attempted


def test_ensure_dataset_rejects_a_corrupt_download(tmp_path, fake_archive):
    """A checksum mismatch raises, and leaves neither the archive nor a partial directory."""
    archive, _ = fake_archive
    dataset = TTildeDataset("T_tildes_tiny", archive.as_uri(), "0" * 64)
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ensure_dataset(dataset, data_dir)
    assert not (data_dir / "T_tildes_tiny.tar.gz").exists()
    assert not (data_dir / "T_tildes_tiny").exists()


def test_ensure_dataset_uses_an_archive_already_in_place(tmp_path, fake_archive):
    """An archive already in the data directory is verified and extracted, but not deleted.

    It is the user's file (e.g. downloaded by hand), not ours to remove.
    """
    archive, sha = fake_archive
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    local = data_dir / "T_tildes_tiny.tar.gz"
    local.write_bytes(archive.read_bytes())
    dataset = TTildeDataset("T_tildes_tiny", "file:///does/not/exist", sha)

    target = ensure_dataset(dataset, data_dir)

    assert (target / "T_tilde_2_00").is_dir()
    assert local.exists()  # not ours to delete


# ---------------------------------------------------------------------------
# the real dataset (skipped when it has not been downloaded)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_DATASET.is_dir(), reason="T-tilde dataset not downloaded")
def test_real_dataset_matches_the_grid(blast_grid):
    """The real dataset (skipped if not downloaded) loads with the default grid.

    Checks the shape against NonLimberGrid and that every value is finite.
    """
    store = TTildeStore(REAL_DATASET, grid=blast_grid)
    matrix = store.get("T_2_00")
    assert matrix.shape == blast_grid.tilde_shape
    assert bool(jnp.all(jnp.isfinite(matrix)))


@pytest.mark.skipif(not REAL_DATASET.is_dir(), reason="T-tilde dataset not downloaded")
def test_real_dataset_has_the_eight_matrices_the_rule_needs():
    """The real dataset (skipped if not downloaded) contains the eight matrices.

    The same set t_tilde_names can ask for.
    """
    names = [
        "T_2_00",
        "T_0_00",
        "T_minus2_00",
        "T_0_02",
        "T_0_20",
        "T_2_02",
        "T_2_20",
        "T_2_22",
    ]
    for name in names:
        assert (REAL_DATASET / directory_name(name)).is_dir(), name
