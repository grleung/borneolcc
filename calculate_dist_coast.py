"""
This script calculates low-level moisture flux convergence and cloud fraction as a function of
hour of the day and distance from the coast. Output is used for generating Figure 9 in Leung and van den Heever (2025);
see paper_figures.ipynb for plotting.
"""

# importing files
import numpy as np
import pandas as pd
import xarray as xr
import dask.distributed as dd

client = dd.Client("solvarg:8788")  # my dask scheduler

client.upload_file("shared_model_params.py")
from shared_model_params import get_rams_output, p00, cp, rd

client.upload_file("shared_processing.py")
from shared_processing import find_paths_in_time_range


runs = ["lc1960", "lc2019"]

time_resolution = 1  # time resolution in hours (1 hr)
times = np.arange(
    8, 20, time_resolution
)  # times to check as hour of the day (localtime)

horizontal_resolution = 0.6  # horizontal spatial resolution in km (600m = 4dx)
dists = np.arange(
    0, 60, 0.6
)  # distances from coastline to use as binning variable (in km)


def compute_ll_moistureconv(ds, zslice=slice(1, 17)):
    """
    Calculates the low level moisture convergence integrated vertically over selected levels.
    By default, integrates from the first model level above the surface to level 17 ~ 0-1km.

    """
    from shared_model_params import alt, dz, remove_boundaries, dx

    # remove points outside analysis boundaries
    ds = remove_boundaries(ds)

    # assign altitude and dz coordinates in m
    ds = ds.assign_coords(alt=("z", alt / 1000))
    ds = ds.assign_coords(dz=("z", dz))

    # select only the z levels specified
    ds = ds.sel(z=zslice)

    # preliminary calculations of pressure (hPa), temp (K), and density (kg/m3)
    ds = ds.assign(
        {
            "PRES": (p00 * (ds.PI / cp) ** (cp / rd)) / 100,
            "TEMP": ds.THETA * (ds.PI / cp),
        }
    )
    ds = ds.assign(DENS=(100 * ds.PRES) / (rd * ds.TEMP * (1 + (0.61 * ds.RV))))

    # calculate advective contribution to vertical integral of moisture flux convergence
    # vertical integral of (-U * d(r*rho)/dx - V* d(r*rho)/dy),
    # where r/RV is mixing ratio of vapor
    # note dx and dy are equal in our simulation so we just use dx here
    # also note that MFC is typically in units of kg kg^(-1) s^(-1) but we are integrating
    # vertically so we also weight by density; this means we get something with final units
    # kg m^(-2) s^(-1)
    ds = ds.assign(
        MFC_adv=-(
            (
                (ds.UP * ((ds.RV * ds.DENS).differentiate("x") / dx))
                + (ds.VP * ((ds.RV * ds.DENS).differentiate("y") / dx))
            )
            * ds.dz
        ).sum(dim="z")
    )

    # calculate convergent contribution to vertical integral of moisture flux convergence:
    # vertical integral of (-r*rho * dU/dx - r*rho*dV/dy )
    ds = ds.assign(
        MFC_conv=-(
            (
                (ds.RV * ds.DENS * (ds.UP.differentiate("x") / dx))
                + (ds.RV * ds.DENS * (ds.VP.differentiate("y") / dx))
            )
            * ds.dz
        ).sum(dim="z")
    )
    ds = ds.assign(MFC=ds.MFC_adv + ds.MFC_conv)

    return ds.MFC


def compute_cfrac(path, run):
    """
    Calculates the cloud fraction at each grid cell (as 0 or 1) based on tobac-derived condensate masks.
    """

    from shared_model_params import bxy, ny, nx

    time = pd.to_datetime(path.split("/")[-1][4:-6])

    # read features from tobac
    features = pd.read_parquet(
        f"/squall/gleung/borneolcc-analysis/tobac/{run}_rte/qc_final_cloudy_updrafts.pq"
    )
    features = features[features.cond_ncells > 0]
    # pick out only features from time we are looking at
    sub = features[features.time == time]

    # read in tobac cloud mask
    mask = xr.open_dataarray(
        f"/squall/gleung/borneolcc-analysis/tobac/{run}_rte/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
        engine="h5netcdf",
        chunks="auto",
    )
    # remove boundaries outside of analysis area
    mask = mask.sel(x=slice(bxy, nx - bxy - 1), y=slice(bxy, ny - bxy - 1))

    # if a valid feature mask is anywhere in the column, assign cloud fraction of 1
    mask = mask.isin(sub.feature.unique()).sum(dim="z")
    mask = mask >= 1

    return mask


def compute_mean_bydistcoast(ds, bins=dists):
    """
    Calculates the mean of the given xarray (ds) as a function of distance from the coastline.

    """

    from scipy.ndimage import distance_transform_edt
    from xarray.groupers import BinGrouper
    from shared_model_params import landmask, dx

    # calculate the distance from the coastline using distance transform from scipy
    # land points are those with landmask > 0 over a rolling 20x20 gridpoint window
    mask = landmask.rolling(x=20, y=20, center=True, min_periods=1).mean() > 0
    dist_coast = distance_transform_edt(mask)

    ds = ds.to_dataset()

    # set the distance from coast as a variable in dataset
    ds = ds.assign(dist_coast=(("y", "x"), dx * dist_coast / 1000))

    # group by specified distance bins and take the mean value
    # bins are labelled by center point of each bin for plotting purposes
    out = ds.groupby(
        dist_coast=BinGrouper(bins=bins, labels=(bins[1:] + bins[:-1]) / 2)
    ).mean()

    del ds

    return out


# loop over runs
for run in runs:
    print(run)

    mfc = []
    cf = []

    # loop over times
    for t in times:
        print(t)

        # for given hour of the day, find RAMS paths within the time resolution we want
        paths = find_paths_in_time_range(run, t, time_resolution)

        # computing moisture convergence for times at this hour of the day
        m = client.map(
            get_rams_output,
            paths,
            variables=["RV", "UP", "VP", "PI", "THETA", "RV"],
        )
        m = client.map(compute_ll_moistureconv, m)
        m = client.gather(m)
        m = xr.concat(m, dim="time").mean(dim="time")
        m = compute_mean_bydistcoast(m).MFC.compute()
        mfc.append(m)

        # computing cloud fraction for times at this hour of the day
        c = client.map(compute_cfrac, paths, run=run)
        c = client.gather(c)
        c = xr.concat(c, dim="time").mean(dim="time")
        c = compute_mean_bydistcoast(c).compute()
        cf.append(c)

    # concatenate over all hours of the day and save
    mfc = xr.concat(mfc, dim=pd.Series(times, name="hour_day"))
    mfc.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/llmfc-diurnal-{run}.h5",
        engine="h5netcdf",
    )

    cf = xr.concat(cf, dim=pd.Series(times, name="hour_day"))
    cf.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/cldfrac-diurnal-{run}.h5",
        engine="h5netcdf",
    )
