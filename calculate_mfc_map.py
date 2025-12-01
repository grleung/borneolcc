"""
This creates analysis plots for Figure 8. It calculates the mean spatial map of surface flux and near-surface thetav
for given time windows.
"""

# importing files
import numpy as np
import pandas as pd
import xarray as xr
import dask.distributed as dd

client = dd.Client("snowfall1:8786")  # my dask scheduler

client.upload_file("shared_model_params.py")
from shared_model_params import (
    get_rams_output,
    landmask,
    remove_boundaries,
    topo,
)

client.upload_file("shared_processing.py")
from shared_processing import (
    find_paths_in_time_range,
    compute_ll_moistureconv,
)


topo = topo.TOPT.compute()
landmask = landmask.compute()

runs = ["lc1960", "lc2019"]

time_resolution = 0.5  # time resolution in hours (30mins)
times = [6,7,8, 9, 10, 11, 12, 13,14,15,16,17,18,19]

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
            variables=["RV", "UP", "VP", "PI", "THETA", "RV"],
        )

        # exclude points near boundaries
        ds = client.map(remove_boundaries, ds)

        # compute moisture convergence
        ds = client.map(compute_ll_moistureconv, ds, landmask=landmask,topo=topo)

        ds = client.gather(ds)

        # take mean value over the times in paths
        # assign that mean to the output xarray
        out.append(xr.concat(ds, dim="time").mean(dim="time").compute())

    out = xr.concat(out, dim=pd.Series(times, name="hour_day"))

    out.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/llmfc-maps-diurnal-{run}.h5",
        engine="h5netcdf",
    )
