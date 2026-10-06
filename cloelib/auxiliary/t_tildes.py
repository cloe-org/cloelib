r"""T-tilde basis integrals for the beyond-Limber angular power spectra.

The $\tilde{T}$ are cosmology-independent integrals of two spherical Bessel
functions (or their second derivatives) against the Chebyshev polynomials
that expand the matter power spectrum in $\log k$,

$$
\tilde{T}(\ell, \chi, R, n) = \int \mathrm{d}k\, k^{\beta}\,
j_\ell^{(A)}(k\chi)\, j_\ell^{(B)}(kR\chi)\, T_n(\log_{10} k),
$$

so they are computed once, for a fixed set of grids, and distributed as a
dataset (like the emulator weights). This module holds

- `NonLimberGrid`: the grids a dataset was computed on. Nothing in the
  beyond-Limber computation may use other grids, so they are *not* user
  parameters: the integration weights and the evaluation nodes all come from
  here.
- `ensure_dataset`: download (once, with checksum verification) and extract
  a dataset next to this module, in `t-tildes-data/`.
- `TTildeStore`: lazy access to the matrices of a dataset, one per name
  (`T_2_00`, `T_minus2_00`, ...), as arrays of shape
  `(n_ell, n_chi, n_R, n_cheb)`.

The default dataset is the one of Chiarenza et al., arXiv:2609.01855
(`Blast.jl`), Zenodo record 21717794.
"""

import hashlib
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from cloelib.auxiliary.chebyshev import (
    chebyshev_points_interval,
    clenshaws_curtis_quadrature,
)
from cloelib.auxiliary.math_utils import simpsons_weights_jit

jax.config.update("jax_enable_x64", True)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "t-tildes-data"


@dataclass(frozen=True, eq=False)
class NonLimberGrid:
    r"""Grids a T-tilde dataset was computed on, and the quadrature weights on them.

    Compared and hashed by identity (it holds arrays).

    Attributes:
      ells: multipoles the T-tilde are tabulated at, shape `(n_ell,)`.
      chi: comoving distances [Mpc], uniformly spaced, shape `(n_chi,)`.
      R: ratios $\chi_2/\chi_1$ in $(0, 1]$, increasing, shape `(n_R,)`.
      k_cheb: Chebyshev nodes in $\log_{10} k$ [$k$ in 1/Mpc], shape
        `(n_cheb,)`, at which the matter power spectrum is sampled.
      w_chi: quadrature weights of the $\chi$ integral, *including* the
        $\chi$ measure factor (`chi * simpson * dchi`), shape `(n_chi,)`.
      w_R: Clenshaw-Curtis weights of the $R$ integral, shape `(n_R,)`.
    """

    ells: Array
    chi: Array
    R: Array
    k_cheb: Array
    w_chi: Array
    w_R: Array

    @property
    def n_ell(self) -> int:
        return len(self.ells)

    @property
    def n_chi(self) -> int:
        return len(self.chi)

    @property
    def n_R(self) -> int:
        return len(self.R)

    @property
    def n_cheb(self) -> int:
        return len(self.k_cheb)

    @property
    def tilde_shape(self) -> tuple[int, int, int, int]:
        """Shape of one T-tilde matrix: `(n_ell, n_chi, n_R, n_cheb)`."""
        return (self.n_ell, self.n_chi, self.n_R, self.n_cheb)

    @classmethod
    def blast_128_64_161(cls) -> "NonLimberGrid":
        r"""The grids of `T_tildes_128_64_161` (the default dataset).

        - $\ell$: the nodes of a 100-point Chebyshev grid on $[2, 2000]$
          below 220 (22 values, increasing).
        - $\chi$: 128 points, uniform on $[26, 7000]$ Mpc.
        - $R$: the 64 positive nodes of a 129-point Clenshaw-Curtis rule on
          $[-1, 1]$, increasing. Its weights follow `Blast.jl`: the weight of
          the smallest $R$ is halved, a numerical correction of the
          truncation to $R > 0$ (not the analytic solution).
        - $k$: 161 Chebyshev nodes in $\log_{10} k$ on $[5\times10^{-5}, 16]$
          1/Mpc.
        """
        n_chi, n_R, n_k = 128, 64, 160

        ells = jnp.flip(chebyshev_points_interval(100, 2.0, 2000.0))
        ells = ells[ells < 220.0]

        chi, d_chi = jnp.linspace(26.0, 7000.0, n_chi, retstep=True)
        w_chi = chi * simpsons_weights_jit(n_chi) * d_chi

        R_all, w_all = clenshaws_curtis_quadrature(2 * n_R + 1, -1.0, 1.0)
        R = jnp.flip(R_all[:n_R])
        w_R = jnp.flip(w_all[:n_R]).at[0].multiply(0.5)

        k_cheb = chebyshev_points_interval(
            n_k, float(np.log10(5e-5)), float(np.log10(16.0))
        )
        return cls(ells=ells, chi=chi, R=R, k_cheb=k_cheb, w_chi=w_chi, w_R=w_R)


@dataclass(frozen=True)
class TTildeDataset:
    """Where to get a T-tilde dataset from.

    Attributes:
      name: name of the directory the archive extracts to.
      url: archive URL (a `.tar.gz`).
      sha256: expected SHA-256 of the archive.
    """

    name: str
    url: str
    sha256: str


BLAST_128_64_161 = TTildeDataset(
    name="T_tildes_128_64_161",
    url="https://zenodo.org/records/21717794/files/T_tildes_128_64_161.tar.gz?download=1",
    sha256="fbb6d4834facc7d3b0195c42718ad58e0651927372de67d1756d32cf72c5489e",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
            return
        root = destination.resolve()
        for member in tar.getmembers():
            if not (root / member.name).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(destination)


def ensure_dataset(
    dataset: TTildeDataset = BLAST_128_64_161,
    data_dir: str | os.PathLike | None = None,
) -> Path:
    """Directory of `dataset`, downloaded and extracted first if missing.

    Nothing is downloaded when the directory already exists. Otherwise the
    archive is fetched into `data_dir` (default: `t-tildes-data/` next to
    this module), its SHA-256 checked against the expected one, and
    extracted; an archive already present in `data_dir` is used as is (but
    checked all the same). An archive downloaded here is removed after
    extraction.

    Returns:
      Path: the directory holding the `T_tilde_*` folders.
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    target = data_dir / dataset.name
    if target.is_dir() and any(target.iterdir()):
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{dataset.name}.tar.gz"
    downloaded = not archive.exists()
    if downloaded:
        print(f"Downloading {dataset.name} (large file) from {dataset.url} ...")
        urllib.request.urlretrieve(dataset.url, archive)

    checksum = _sha256(archive)
    if checksum != dataset.sha256:
        if downloaded:
            archive.unlink()
        raise ValueError(
            f"Checksum mismatch for {archive}: expected {dataset.sha256}, got "
            f"{checksum}."
        )

    _extract(archive, data_dir)
    if downloaded:
        archive.unlink()
    if not target.is_dir():
        raise FileNotFoundError(
            f"Archive {archive} did not extract to the expected {target}."
        )
    return target


_NAME = re.compile(r"^T_(minus)?(\d+)_(\d{2})$")
_ELL_FILE = re.compile(r"^T_tilde_l_(\d+(?:\.\d+)?)\.npy$")
# Files carry the multipole rounded to one decimal.
_ELL_FILE_TOLERANCE = 0.051


def directory_name(name: str) -> str:
    """Directory of a matrix in a dataset: `T_minus2_00` -> `T_tilde_-2_00`."""
    match = _NAME.match(name)
    if match is None:
        raise ValueError(
            f"Invalid T-tilde name {name!r}: expected e.g. 'T_2_00', 'T_minus2_00'."
        )
    minus, power, derivatives = match.groups()
    return f"T_tilde_{'-' if minus else ''}{power}_{derivatives}"


class TTildeStore:
    """Lazy access to the T-tilde matrices of a dataset.

    Each matrix is about 230 MB, so one is read from disk only when first
    asked for (`get`) and then kept: a computation only loads the matrices
    its contribution pairs need.

    Args:
      path: directory of an already extracted dataset (holding the
        `T_tilde_*` folders). By default the standard dataset is used,
        downloaded first if needed (`ensure_dataset`).
      grid: grids the dataset was computed on. Defaults to those of the
        standard dataset (`NonLimberGrid.blast_128_64_161`).
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        grid: NonLimberGrid | None = None,
    ) -> None:
        self.grid = grid if grid is not None else NonLimberGrid.blast_128_64_161()
        self.path = Path(path) if path is not None else ensure_dataset()
        if not self.path.is_dir():
            raise FileNotFoundError(f"T-tilde dataset not found: {self.path}")
        self._cache: dict[str, Array] = {}

    def get(self, name: str) -> Array:
        """The matrix `name` (e.g. `"T_minus2_00"`), shape `grid.tilde_shape`."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _ell_files(self, folder: Path) -> dict[float, Path]:
        files = {}
        for entry in folder.iterdir():
            match = _ELL_FILE.match(entry.name)
            if match is not None:
                files[float(match.group(1))] = entry
        return files

    def _load(self, name: str) -> Array:
        folder = self.path / directory_name(name)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing T-tilde directory: {folder}")

        files = self._ell_files(folder)
        grid = self.grid
        expected = (1, grid.n_chi, grid.n_R, grid.n_cheb)
        blocks = []
        for ell in np.asarray(grid.ells):
            match = [f for f in files if abs(f - ell) <= _ELL_FILE_TOLERANCE]
            if len(match) != 1:
                raise FileNotFoundError(
                    f"{folder}: expected one file for ell = {ell:.3f}, found "
                    f"{len(match)} (available: {sorted(files)})."
                )
            block = np.load(files[match[0]], allow_pickle=False)
            if block.shape != expected:
                raise ValueError(
                    f"{files[match[0]]}: shape {block.shape}, expected {expected}."
                )
            blocks.append(block[0])
        return jnp.asarray(np.stack(blocks))
