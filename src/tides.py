import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pyTMD
from pyTMD.compute import tide_elevations
from scipy.ndimage import binary_erosion
from timescale.time import convert_datetime

logger = logging.getLogger(__name__)


def find_nearest_coordinate(latlon: np.ndarray, mask: np.ndarray, yi: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Find the closest coordinate from a mask of valid coordinates.

    Parameters:
    - latlon (ndarray): Target latitude, longitude as list (N, 2)
    - mask (ndarray): Mask of valid pixels
    - yi (ndarray): Y coordinates cooresponding to mask
    - xi (ndarray): X coordinates cooresponding to mask

    Returns:
    - nearest_latlon: The closest valid coordinates as a list
    """
    eroded_mask = binary_erosion(mask.astype(np.uint8), np.ones((3, 3))) == 1
    # Convert mask to coords
    vy, vx = np.where(eroded_mask)
    valid_latlon = np.array(list(zip(yi[vy], xi[vx])))

    # Compute Euclidean distances (simplified for small distances)
    diff = valid_latlon[:, np.newaxis, :] - latlon[np.newaxis, :, :]
    distances = np.sqrt(np.sum(np.power(diff, 2), axis=2))

    # Find the index of the closest coordinate
    nearest_idx = np.argmin(distances, axis=0)

    nearest_latlon = valid_latlon[nearest_idx]
    return nearest_latlon


def datetimes_to_delta(ts: list[datetime]) -> np.ndarray:
    return convert_datetime(np.array([np.datetime64(t) for t in ts]))


class TideModel:
    def __init__(self, model: pyTMD.io.model, model_directory: Path, model_name: str) -> None:
        self.model = model
        self.model_directory = model_directory
        self.model_name = model_name

        assert model.model_file is not None

        hc, xi, yi, c = pyTMD.io.GOT.read_ascii_file(model.model_file[0], compressed=model.compressed)  # type: ignore
        self.hc = hc
        self.xi = xi
        self.yi = yi
        self.c = c
        # invert tidal constituent mask
        self.mz = np.invert(hc.mask)

    def tide_elevations(self, latlon: np.ndarray, times: list[datetime], samples: int = 10) -> np.ndarray:
        """_summary_

        Args:
            latlon (np.ndarray): Target latitude, longitude as list (N, 2)
            times (list[datetime]): List of datetimes to process (N)
            samples (int, optional): Number of intorpolation samples. Defaults to 10.

        Returns:
            np.ndarray: The tidal height for each latlon/time in meters.
        """
        latlon = latlon.astype(np.float64)
        lt1 = np.nonzero(latlon[:, 1] < 0)
        latlon[:, 1][lt1] += 360.0

        latlon_close = find_nearest_coordinate(latlon, self.mz, self.yi, self.xi)
        yxs = np.linspace(latlon, 2 * latlon_close - latlon, samples)
        S, N, _ = yxs.shape
        ys = yxs[:, :, 0].flatten()
        xs = yxs[:, :, 1].flatten()
        deltas = datetimes_to_delta(times)
        deltas_samples = np.repeat(deltas[np.newaxis, :], samples, axis=0)
        flat_deltas_samples = deltas_samples.flatten()

        flat_elevations = tide_elevations(
            xs, ys, delta_time=flat_deltas_samples, DIRECTORY=self.model_directory, MODEL=self.model_name
        )

        elevations = flat_elevations.reshape((S, N))

        out_elevations = []
        for n in range(N):
            si = np.where(~elevations.mask[:, n])[0][0]
            out_elevations.append(elevations.data[si, n])

        return np.array(out_elevations)


def tide_model(model_directory: Path | None, model_name: str, model_format: str) -> TideModel | None:
    if model_directory is None or not model_directory.exists():
        logger.warning(f"Invalid tides directory {model_directory}. Skipping...")
        return None

    assert model_name == "GOT4.10", "Only support GOT4.10 for now!"

    # Load the GOT4.10c model
    model = pyTMD.io.model(model_directory, format=model_format).elevation(model_name)

    return TideModel(model, model_directory, model_name)
