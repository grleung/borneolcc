"""
This creates analysis plots for Figure 8. It calculates the mean spatial map of surface flux and near-surface thetav
for given time windows.
"""

# importing files
import numpy as np
import pandas as pd
import xarray as xr
import dask.distributed as dd

client = dd.Client("snowfall2:8786")  # my dask scheduler

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
    compute_surf_pert,
)


topo = topo.TOPT.compute()
landmask = landmask.compute()

runs = ["lc1960", "lc2019"]

time_resolution = 0.5  # time resolution in hours (30mins)
times = [8, 9, 10, 11, 12, 13]

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
            variables=["SFLUX_T", "SFLUX_R", "THETA", "RV"],
        )

        # exclude points near boundaries
        ds = client.map(remove_boundaries, ds)

        # compute energy budget
        ds = client.map(compute_surf_pert, ds, landmask=landmask, topo=topo)

        ds = client.gather(ds)

        # take mean value over the times in paths
        # assign that mean to the output xarray
        out.append(xr.concat(ds, dim="time").mean(dim="time").compute())

    out = xr.concat(out, dim=pd.Series(times, name="hour_day"))

    out.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/maps-diurnal-nearsurf75-{run}.h5",
        engine="h5netcdf",
    )
