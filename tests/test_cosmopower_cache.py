import importlib.util
import sys
import types
import urllib.request
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "cloelib" / "cosmology" / "cosmopower_jax_cosmology.py"
)


def load_module(monkeypatch):
    cosmology = types.ModuleType("cloelib.cosmology.cosmology")
    cosmology.Background = object
    cosmology.Perturbations = object
    extrapolator = types.ModuleType("cloelib.auxiliary.extrapolator")
    extrapolator.extend_spectra = lambda *args: args
    monkeypatch.setitem(sys.modules, cosmology.__name__, cosmology)
    monkeypatch.setitem(sys.modules, extrapolator.__name__, extrapolator)

    spec = importlib.util.spec_from_file_location("cosmopower_cache_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_stores_cosmopower_artifacts_in_configured_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    downloads = []

    def download(url, filename):
        path = Path(filename)
        path.touch()
        downloads.append(path)

    monkeypatch.setenv("CLOELIB_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    load_module(monkeypatch)

    assert downloads == [
        cache_dir / "cosmopower-jax" / "k-modes.txt",
        cache_dir / "cosmopower-jax" / "curvature-kmodes.txt",
    ]


def test_cosmopower_cache_uses_xdg_then_user_cache(monkeypatch, tmp_path):
    xdg_cache = tmp_path / "xdg"
    monkeypatch.delenv("CLOELIB_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda url, filename: Path(filename).touch(),
    )

    module = load_module(monkeypatch)

    assert Path(module.emulator_data("warm.npz")) == (
        xdg_cache / "cloelib" / "cosmopower-jax" / "warm.npz"
    )

    home = tmp_path / "home"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(home))

    module = load_module(monkeypatch)

    assert Path(module.emulator_data("warm.npz")) == (
        home / ".cache" / "cloelib" / "cosmopower-jax" / "warm.npz"
    )


def test_cosmopower_warm_cache_does_not_download(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    artifact_dir = cache_dir / "cosmopower-jax"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "k-modes.txt").touch()
    (artifact_dir / "curvature-kmodes.txt").touch()
    artifact = artifact_dir / "warm.npz"
    artifact.touch()

    monkeypatch.setenv("CLOELIB_CACHE_DIR", str(cache_dir))

    def fail_download(*args):
        raise AssertionError("unexpected download")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_download)

    module = load_module(monkeypatch)

    assert Path(module.emulator_data("warm.npz")) == artifact
