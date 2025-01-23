import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy
import cartopy.crs as ccrs
import datetime as dt
import dask
import dask.distributed as dd
import glob
import matplotlib as mpl
from netCDF4 import Dataset
from satpy import Scene, find_files_and_readers
from satpy.resample import get_area_def
import datetime as dt
import s3fs
import himawari_api as hapi

mpl.use("Agg")

client = dd.Client("snowfall1:8786")
client.upload_file("shared_model_params.py")
client.upload_file("shared_plotting.py")

from shared_plotting import *
from shared_model_params import (
    get_rams_output,
    rams_dims_lite,
    assign_dz,
    zt,
    dz,
    p00,
    cp,
    rd,
    ana_var,
    lite_var,
)

lcver = "lc2019"
rver = "rte"

dataPath = f"/squall/gleung/borneolcc/{lcver}/{rver}/"
grid = "g1"
figPath = f"/squall/gleung/borneolcc-figures/pres-{lcver}-{rver}/"

if not os.path.isdir(figPath):
    os.mkdir(figPath)

all_paths = [
    p
    for p in sorted(glob.glob(f"{dataPath}/a-L-*000-g1.h5"))
    if (f"{p.split('/')[-1][4:-8]}.png" not in os.listdir(figPath))
    and (
        pd.to_datetime(p.split("/")[-1][4:-8])
        >= pd.to_datetime("2019-09-17-0400")
    )
]

print(len(all_paths), all_paths)

# these 2d things are fixed in time
coord = xr.open_dataset(
    f"/squall/gleung/borneolcc/{lcver}/{rver}/a-A-2019-09-16-140000-g1.h5",
    drop_variables=[v for v in ana_var if v not in ["TOPT", "GLAT", "GLON"]],
    engine="h5netcdf",
    chunks="auto",
    phony_dims="access",
).rename_dims({"phony_dim_0": "y", "phony_dim_1": "x"})

drop_var = [
    v
    for v in lite_var
    if v
    not in [
        "PCPRR",
        "RCP",
        "RPP",
        "RSP",
    ]
]


def add_dz(ds):
    ds = ds.assign(dz=("z", dz))
    return ds


def add_variables(ds):
    ds = ds.assign(
        {
            "pcp": ds.PCPRR * 3600,
            "tcon": ds.RCP + ds.RPP + ds.RSP,
        }
    )

    ds = ds.assign(intcon=calculate_intcon(ds.tcon, ds.dz))

    ds = ds[["pcp", "intcon"]]
    return ds


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})


def calculate_intcon(tcon, dz):
    intcon = ((1000 / 997) * (tcon * dz).sum(dim="z")) + 1e-5

    return intcon


def plot_quicklook(ds, time, ahipaths, glon, glat, tracks, savePath):
    figPath = f"{savePath}/{time.strftime('%Y-%m-%d-%H%M')}.png"

    latmin = glat.min()
    latmax = glat.max()
    lonmin = glon.min()
    lonmax = glon.max()

    intcon = ds.intcon
    pcp = ds.pcp

    # reading in and plotting the satellite data from AHI
    scn = Scene(filenames=ahipaths, reader="ahi_hsd")
    scn.load(["B01", "B02", "B03", "B04"])
    scn = scn.resample(resampler="native")
    scn = scn.crop(ll_bbox=(108.65, -0.164, 111.61, 2.85))
    scn = scn.resample(resampler="native")

    scn.load(["true_color"])
    crs = scn["true_color"].attrs["area"].to_cartopy_crs()

    tc = scn.show("true_color")

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(10, 5),
        subplot_kw={"projection": crs},
    )
    axes = axes.flatten()

    axes[0].imshow(
        tc.pil_image(), transform=crs, extent=crs.bounds, origin="upper"
    )

    c = axes[1].pcolormesh(
        glon,
        glat,
        intcon,
        cmap=cloud,
        norm=mcolors.LogNorm(vmin=1e-3, vmax=1e2),
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(c, ax=axes[1], label="mm")

    axes[0].set_title("(a) AHI True Color")
    axes[1].set_title("(b) Max Vertical Velocity")

    for ax in axes:
        ax.coastlines()
        add_latlon(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plot_tracks = tracks[tracks.time == time]

    for c in plot_tracks.cell.unique():
        p = tracks[tracks.cell == c]
        axes[1].plot(p.lon, p.lat, zorder=1, lw=0.5, color="white")

    axes[1].scatter(
        plot_tracks.lon,
        plot_tracks.lat,
        s=5,
        zorder=2,
        edgecolors="white",
        linewidths=0.5,
        facecolor="None",
    )

    plt.suptitle(
        f"{time.strftime('%Y-%m-%d %H:%M')} UTC ({(time+dt.timedelta(hours=8)).strftime('%H:%M')} LT)",
        fontsize=30,
    )
    plt.tight_layout()

    plt.savefig(figPath)
    plt.close(fig)


satpaths_dict = hapi.group_files(
    glob.glob("/squall/gleung/ahi/HIMAWARI-8/AHI-L1b-FLDK/*/*/*/*/*"),
    key="start_time",
)

glat = coord.GLAT.compute()
glon = coord.GLON.compute()


tracks = pd.read_parquet(
    f"/squall/gleung/borneolcc-analysis/tobac/{lcver}_rte/final_cond-w_segmented_tracks.pq"
)

for paths in np.array_split(sorted(all_paths), len(all_paths) // 15):
    print(paths)
    times = [
        pd.to_datetime(p.split("/")[-1][4:-6])
        for p in paths
        if (pd.to_datetime(p.split("/")[-1][4:-6]) in satpaths_dict.keys())
    ]
    satpaths = [
        satpaths_dict.get(time)
        for time in times
        if (time in satpaths_dict.keys())
    ]
    print(times)

    try:
        ds = client.map(
            get_rams_output,
            [p for p in paths],
            variables=["RCP", "RSP", "RPP", "PCPRR"],
            prep_tobac=True,
        )

        ds = client.map(add_dz, ds)

        ds = client.map(add_variables, ds)

        out = client.map(
            plot_quicklook,
            ds,
            times,
            satpaths,
            [glon.values] * len(times),
            [glat.values] * len(times),
            [tracks] * len(times),
            savePath=f"/squall/gleung/borneolcc-figures/pres-{lcver}-{rver}/",
        )
        out = client.gather(out)
    except:
        print(times)
        pass
