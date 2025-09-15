# libraries shared across most of these plots

import numpy as np
import pandas as pd
import xarray as xr
import dask.distributed as dd
import metpy.calc as mpcalc
import metpy.units as units

client = dd.Client("solvarg:8786")

client.upload_file("shared_model_params.py")
from shared_model_params import (
    get_rams_output,
    landmask,
    remove_boundaries,
    cp,
    rd,
    p00,
)

client.upload_file("shared_processing.py")
from shared_processing import (
    find_paths_in_time_range,
)

landmask = landmask.compute()


def run_thickness(pres_1, pres_2, t_1, t_2):
    height = mpcalc.thickness_hydrostatic(
        [pres_1, pres_2] * units.units("hPa"),
        [t_1, t_2] * units.units("celsius"),
    ).magnitude

    return height


run_thickness = np.vectorize(run_thickness)


def compute_lcl(p, t, td):
    # lcl calculation
    lcl_p, lcl_t = mpcalc.lcl(
        p * units.units("hPa"),
        t * units.units("celsius"),
        td * units.units("celsius"),
    )

    height = run_thickness(
        p * units.units("hPa"), lcl_p, t * units.units("celsius"), lcl_t
    )
    return height


def compute_lfc(p, t, td):
    lfc_p, lfc_t = mpcalc.lfc(
        [p] * units.units("hPa"),
        [t] * units.units("celsius"),
        [td] * units.units("celsius"),
    )

    height = run_thickness(
        p[0] * units.units("hPa"), lfc_p, t[0] * units.units("celsius"), lfc_t
    )

    return height


compute_lfc = np.vectorize(compute_lfc, signature="(z),(z),(z)->()")


def calculate_mean_lcl_lfc(ds):

    ds = ds.where(landmask)

    # need to coarsen for computational efficiency
    ds = ds.coarsen(x=300, y=300, boundary="pad").mean()

    # precalculate needed variables for metpy calculation
    ds = ds.assign(
        {
            "T": (ds.THETA * (ds.PI / cp)) - 273.15,  # temperature in degC
            "PRES": (p00 * (ds.PI / cp) ** (cp / rd)) / 100,  # pressure in hPa
        }
    )
    ds = ds.assign(
        Td=(
            ("z", "y", "x"),
            mpcalc.dewpoint_from_specific_humidity(
                ds.PRES.data * units.units("hPa"),
                ds.T.data * units.units("celsius"),
                ds.RV.data * units.units("kg/kg"),
            ).magnitude,
        )
    )  # dewpoint in degC

    ds = ds[["PRES", "T", "Td"]]

    # stack x and y and drop all points that are over ocean
    ds = ds.stack(xy=("x", "y"))
    ds = ds.dropna(dim="xy", how="any")
    ds = ds.transpose("xy", "z")

    # compute LCL
    ds = ds.assign(
        LCL_height=(
            "xy",
            compute_lcl(ds["PRES"].sel(z=1), ds["T"].sel(z=1), ds["Td"].sel(z=1)),
        )
    )

    # compute LFC
    ds = ds.assign(LFC_height=("xy", compute_lfc(ds["PRES"], ds["T"], ds["Td"])))

    # take mean over land points only
    ds = ds[["LCL_height", "LFC_height"]].mean()

    return ds


runs = ["lc2019"]

time_resolution = 0.5  # time resolution in hours (30mins)
times = np.arange(
    0,
    24,
    time_resolution,
   
)  # times to check as hour of the day (localtime)

for run in runs:

    out = []

    for t in times:
        paths = find_paths_in_time_range(run, t, time_resolution)

        # read rams output
        ds = client.map(
            get_rams_output,
            paths,
            variables=["PI", "THETA", "RV"],
        )

        # get rid of points outside analysis area
        ds = client.map(remove_boundaries, ds)

        # compute mean over land points
        heights = client.map(calculate_mean_lcl_lfc, ds)

        # gather all analysis files for given time range
        heights = client.gather(heights)
        out.append(xr.concat(heights, dim="time").mean(dim="time").compute())

        print(t)

    # concatenate along hour of day
    out = xr.concat(out, dim=pd.Series(times, name="hour_day"))

    # save output file
    out.to_netcdf(
        f"/squall/gleung/borneolcc-analysis/paper-analysis/lcl-lfc-diurnal-{run}.h5",
        engine="h5netcdf",
        compute=True,
    )
