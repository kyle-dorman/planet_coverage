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

    def tide_elevations(self, latlon: np.ndarray, times: list[list[datetime]], samples: int = 10) -> list[np.ndarray]:
        """Computes tidal elevations for N latlon pairs and N, m times. Where m is variable.

        Args:
            latlon (np.ndarray): Target latitude, longitude as list (N, 2)
            times (listlist[[datetime]]): List of datetimes to process (M, m)
            samples (int, optional): Number of intorpolation samples. Defaults to 10.

        Returns:
            np.ndarray: The tidal height for each latlon/time in meters.
        """
        yxs = self.find_best_coordinates(latlon, samples)

        outs = []
        # Compute tidal elevations per coordinate
        for (y, x), tlist in zip(yxs, times):
            # compute elevations
            elev = tide_elevations(
                x,
                y,
                delta_time=tlist,  # type: ignore
                DIRECTORY=self.model_directory,
                MODEL=self.model_name,
                TYPE="time series",
                TIME="datetime",
            )[0]
            assert not elev.mask.any()
            outs.append(elev.data)
        return outs

    def find_best_coordinates(self, latlon: np.ndarray, samples: int = 10) -> np.ndarray:
        latlon = latlon.astype(np.float64)
        if len(latlon.shape) == 1:
            latlon = latlon[None]
        lt1 = np.nonzero(latlon[:, 1] < 0)
        latlon[:, 1][lt1] += 360.0

        latlon_close = find_nearest_coordinate(latlon, self.mz, self.yi, self.xi)
        yxs = np.linspace(latlon, 2 * latlon_close - latlon, samples)
        S, N, _ = yxs.shape
        ys = yxs[:, :, 0].flatten()
        xs = yxs[:, :, 1].flatten()

        flat_deltas_samples = datetimes_to_delta([datetime.now()] * len(ys))
        flat_elevations = tide_elevations(
            xs, ys, delta_time=flat_deltas_samples, DIRECTORY=self.model_directory, MODEL=self.model_name
        )
        elevations = flat_elevations.reshape((S, N))

        best_idxes = []
        for n in range(N):
            si = np.where(~elevations.mask[:, n])[0][0]
            best_idxes.append(si)

        # Select the best coordinate for each target
        yxs_best = yxs[best_idxes, np.arange(len(best_idxes))]

        return yxs_best


def tide_model(model_directory: Path | None, model_name: str, model_format: str) -> TideModel | None:
    if model_directory is None or not model_directory.exists():
        logger.warning(f"Invalid tides directory {model_directory}. Skipping...")
        return None

    assert model_name == "GOT4.10", "Only support GOT4.10 for now!"

    # Load the GOT4.10c model
    model = pyTMD.io.model(model_directory, format=model_format).elevation(model_name)

    return TideModel(model, model_directory, model_name)
