import logging
from pathlib import Path

import numpy as np
import pyTMD
from pyTMD.compute import tide_elevations

logger = logging.getLogger(__name__)


def calc_tide_elevatons(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    model_directory: Path | None,
    model_name: str,
    model_format: str,
) -> np.ndarray:
    """
    Compute tidal elevations for multiple spatial points over a shared time series
    using a pyTMD tidal model.

    Parameters
    ----------
    xs : np.ndarray
        Array of x-coordinates (typically longitudes) of shape (N,).
        Coordinate reference system must match the model after transformation.
    ys : np.ndarray
        Array of y-coordinates (typically latitudes) of shape (N,).
    ts : np.ndarray
        Array of datetime objects or timestamps representing the time series
        at which to compute tidal elevations. Shape (T,).
    model_directory : Path | None
        Path to the directory containing the pyTMD tidal model files.
        If None, pyTMD will attempt to use default model paths.
    model_name : str
        Name of the tidal model (e.g., "GOT4.10c").
    model_format : str
        Model format string understood by pyTMD (e.g., "netcdf").

    Returns
    -------
    np.ndarray
        Array of tidal elevations with shape (N, T), where:
        - N is the number of spatial points
        - T is the length of the time series
    """

    # Load the tidal model metadata and configuration from disk
    tm = pyTMD.io.model(model_directory, format=model_format).from_database(model_name)

    # Open the tidal elevation dataset ("z" group contains elevation constituents)
    # chunks="auto" enables dask-backed lazy loading for large models
    ds = tm.open_dataset(group="z", chunks="auto")

    # Transform input coordinates into the model's internal coordinate system
    # (e.g., geographic -> projected)
    xs, ys = ds.tmd.transform_as(xs, ys)

    out = []

    # Loop over spatial points; each iteration computes a full time series
    # of tidal elevations for one (x, y) location
    for i in range(len(xs)):
        elev = tide_elevations(
            xs[i],
            ys[i],
            delta_time=ts,  # Time vector for evaluation (shared across points)
            directory=model_directory,
            model=model_name,
            type="time series",  # Indicates we want a full time series output
            standard="datetime",  # Input time format
            method="nearest",  # Nearest-neighbor spatial interpolation
            extrapolate=True,  # Allow extrapolation beyond model domain
            cutoff=np.inf,
        )
        out.append(elev)

    # Stack results into a single array of shape (N, T)
    return np.array(out)
