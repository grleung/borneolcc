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

mpl.use("Agg")

client = dd.Client("tcp://172.16.1.237:8786")
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

lcver = "lc1960"
rver = "rte"

dataPath = f"/squall/gleung/borneolcc/{lcver}/{rver}/"
grid = "g1"
figPath = f"/squall/gleung/borneolcc-figures/{lcver}/{rver}/quicklook/"

if not os.path.isdir(figPath):
    os.mkdir(figPath)

all_paths = [
    p
    for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))
    if f"{p.split('/')[-1][4:-8]}.png" not in os.listdir(figPath)
]

print(all_paths)

# these 2d things are fixed in time
coord = xr.open_dataset(
    f"/squall/gleung/borneolcc/lc1960/{rver}/a-A-2019-09-16-140000-g1.h5",
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
        "WP",
        "PCPRR",
        "PI",
        "THETA",
        "RV",
        "RCP",
        "RPP",
        "RSP",
    ]
]


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})


def calculate_density(pi, theta, rv):
    from shared_model_params import p00, cp, rd

    pres = p00 * (pi / cp) ** (cp / rd)
    temp = theta * (pi / cp)
    dens = pres / (rd * temp * (1 + (0.61 * rv)))

    return dens


def calculate_intcon(tcon, dens, dz):
    intcon = ((1000 / 997) * (tcon * dens * dz).sum(dim="z")) + 1e-5

    return intcon


@dask.delayed
def plot_quicklook(maxw, intcon, pcp, glon, glat, topt, time, figPath):
    latmin = glat.min()
    latmax = glat.max()

    lonmin = glon.min()
    lonmax = glon.max()

    fig, axes = plt.subplots(
        1,
        3,
        sharex=True,
        sharey=True,
        figsize=(19, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    c = axes[0].contourf(
        glon,
        glat,
        maxw,
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=20),
        extend="both",
        levels=np.linspace(-5, 20, 21),
    )
    plt.colorbar(c, ax=axes[0], label="m s$^{-1}$", orientation="horizontal")

    c = axes[1].contourf(
        glon,
        glat,
        intcon,
        cmap=cloud,
        norm=mcolors.LogNorm(),
        extend="both",
        levels=np.logspace(-3, 2, 16),
    )
    plt.colorbar(c, ax=axes[1], label="mm", orientation="horizontal")

    c = axes[2].contourf(
        glon,
        glat,
        pcp,
        norm=mcolors.LogNorm(),
        extend="both",
        levels=np.logspace(0, 2, 11),
    )
    plt.colorbar(c, ax=axes[2], label="mm hr$^{-1}$", orientation="horizontal")

    axes[0].set_title("(a) Max Vertical Velocity")
    axes[1].set_title("(b) Integrated Condensate")
    axes[2].set_title("(c) Precipitation Rate")

    for ax in axes:
        ax.contour(
            glon,
            glat,
            topt,
            levels=np.linspace(0, 4000, 5),
            colors="gray",
            linewidths=0.5,
        )

        add_latlon(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lonmin, lonmax)
        ax.set_ylim(latmin, latmax)

    time = pd.to_datetime(time.t.values)

    plt.suptitle(
        f"{time.strftime('%Y-%m-%d %H:%M')} UTC ({(time+dt.timedelta(hours=8)).strftime('%H:%M')} LT)",
        fontsize=30,
    )

    plt.savefig(f"{figPath}/{time.strftime('%Y-%m-%d-%H%M')}.png")
    plt.close(fig)

    print(f"{figPath}/{time.strftime('%Y-%m-%d-%H%M')}.png")


for paths in np.array_split(all_paths, len(all_paths) // 5):
    times = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]

    try:

        ds = xr.open_mfdataset(
            paths,
            engine="h5netcdf",
            chunks="auto",
            phony_dims="access",
            drop_variables=drop_var,
            combine="nested",
            concat_dim=[pd.Index(times, name="t")],
            parallel=True,
        )

        ds = rename_dims(ds, rams_dims_lite)

        ds = ds.assign(dz=("z", dz))

        maxw = ds.WP.max(dim="z")
        pcp = ds.PCPRR * 3600
        tcon = ds.RCP + ds.RPP + ds.RSP
        dens = calculate_density(ds.PI, ds.THETA, ds.RV)
        intcon = calculate_intcon(tcon, dens, ds.dz)

        del ds

        tasks = [
            plot_quicklook(
                maxw.sel(time=time),
                intcon.sel(time=time),
                pcp.sel(time=time),
                coord.GLON,
                coord.GLAT,
                coord.TOPT,
                time,
                figPath,
            )
            for time in intcon.time
        ]

        dask.compute(tasks)
    except:
        pass
