"""
This script takes RAMS output and creates a hdf5 file (.h5) that contains the
mean thermodynamic profile for the near-surface air and canopy air over land points only,
as an average over the local time diurnal cycle (over the three simulation days). By default,
the time resolution is 30mins. There is one hdf5 file per run, which are saved under
 '/squall/gleung/borneolcc-analysis/paper-analysis/nearsurf-thermo-diurnal-[run].h5'
"""

# importing files
import numpy as np
import pandas as pd
import xarray as xr
import dask.distributed as dd

client = dd.Client("snowfall3:8786")  # my dask scheduler

client.upload_file("shared_model_params.py")
from shared_model_params import get_rams_output, landmask, remove_boundaries

client.upload_file("shared_processing.py")
from shared_processing import (
    find_paths_in_time_range,
    get_land_mean,
    compute_canopy_nearsurf,
)

landmask = landmask.compute()

runs = ["lc1960", "lc2019"]

time_resolution = 0.5  # time resolution in hours (30mins)
times = np.arange(
    0, 24, time_resolution
)  # times to check as hour of the day (localtime)

for run in runs:
    print(run)
    out = []
    for t in times:
        print(t)

        # for given hour of the day, find RAMS paths within the time resolution we want
        paths = find_paths_in_time_range(run, t, time_resolution)

        # read rams output
        ds = client.map(
            get_rams_output,
            paths,
            variables=["THETA", "PI", "RV", "CAN_TEMP", "CAN_RVAP"],
        )

        # exclude points near boundaries
        ds = client.map(remove_boundaries, ds)

        # compute near surface and canopy thermodynamics
        ds = client.map(compute_canopy_nearsurf, ds)

        # compute mean over land points
        ds = client.map(get_land_mean, ds, landmask=landmask)

        ds = client.gather(ds)

        # take mean value over the times in paths
        # assign that mean to the output xarray
        out.append(xr.concat(ds, dim="time").mean(dim="time").compute())

    out = xr.concat(out, dim=pd.Series(times, name="hour_day"))

    out.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/nearsurf-thermo-diurnal-{run}.h5",
        engine="h5netcdf",
    )
