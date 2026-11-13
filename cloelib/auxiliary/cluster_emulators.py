import os
import urllib.request
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight NumPy feed-forward network
# ---------------------------------------------------------------------------


class ClusterEmuNet:
    r"""
    Six-hidden-layer feed-forward network evaluated in NumPy.

    Architecture (halving hidden size at each layer)::

        input  →  fc1 (H)  →  fc2 (H/2)  →  fc3 (H/4)
               →  fc4 (H/8) →  fc5 (H/16) →  fc6 (output)

    Activation: Leaky ReLU (negative slope 0.01) after every hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        output_size: int = 1,
        params_min: np.ndarray = None,
        params_max: np.ndarray = None,
    ):
        # Weight matrices  shape: (out_features, in_features)
        # Bias vectors     shape: (out_features,)
        sizes = [
            (hidden_size, input_size),
            (hidden_size // 2, hidden_size),
            (hidden_size // 4, hidden_size // 2),
            (hidden_size // 8, hidden_size // 4),
            (hidden_size // 16, hidden_size // 8),
            (output_size, hidden_size // 16),
        ]
        self._weights = [np.zeros(s) for s in sizes]
        self._biases = [np.zeros(s[0]) for s in sizes]
        self._x_min = params_min
        self._x_max = params_max

    def load_from_dict(self, weight_dict: dict) -> None:
        r"""
        Load pre-trained weights from a dictionary.

        The dictionary must contain keys ``fc1_w``, ``fc1_b``,
        ``fc2_w``, ``fc2_b``, …, ``fc6_w``, ``fc6_b`` mapping to
        numpy arrays.  This is the primary loading path when weights
        are embedded in :mod:`EmuNetWeights`.

        Parameters
        ----------
        weight_dict : dict
            Dictionary of weight and bias arrays.
        """
        for i in range(1, 7):
            self._weights[i - 1] = weight_dict[f"fc{i}_w"]
            self._biases[i - 1] = weight_dict[f"fc{i}_b"]

    def load_weights(self, npz_path: str) -> None:
        r"""
        Load pre-trained weights from a ``.npz`` file (convenience wrapper).

        Prefer :math:`load_from_dict` with the weights from
        :mod:`EmuNetWeights` for path-independent deployment.

        Parameters
        ----------
        npz_path : str
            Path to the ``.npz`` weight file.
        """
        self.load_from_dict(dict(np.load(npz_path)))

    @staticmethod
    def _leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, 0.01 * x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""
        Forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape ``(N, input_size)``.

        Returns
        -------
        out : np.ndarray
            Output array of shape ``(N, output_size)``.
        """
        out = x

        # Min-max normalisation to [0, 1]
        if self._x_min is not None:
            out = (x - self._x_min) / (self._x_max - self._x_min)

        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            out = out @ W.T + b
            if i < len(self._weights) - 1:  # no activation on output layer
                out = self._leaky_relu(out)
        return out


def get_emulator_data(filename: str, filepath: str, zenodo_url: str = None) -> dict:
    """Download the emulator data file if it does not exist.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    filepath : str
        Local path where the data is or will be saved in
    zenodo_url : str, optional
        The base URL from which to download the file. Defaults to ZENODO_URL.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    # if zenodo_url is None:
    #    zenodo_url = ZENODO_URL

    os.makedirs(filepath, exist_ok=True)
    file_path = os.path.join(filepath, filename)

    if not os.path.exists(file_path):
        url = f"{zenodo_url}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, file_path)

    data = np.load(file_path, allow_pickle=True)

    # convert into dictionary
    data = {key: data[key] for key in data.files}
    data["header"] = data["header"].item()
    data["profile_model"] = data["profile_model"].item()
    if "trunc_fact" in data:
        data["trunc_fact"] = data["trunc_fact"].item()

    return data
