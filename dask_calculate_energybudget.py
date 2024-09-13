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

client = dd.Client("tcp://172.16.1.237:8786")
client.upload_file("shared_model_params.py")
from shared_model_params import (
    get_rams_output,
    rams_dims_lite,
    assign_dz,
    zt,
    dz,
    p00,
    cp,
    rd,
    lv,
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

all_paths = [p for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))]


# these 2d things are fixed in time
coord = xr.open_dataset(
    f"/squall/gleung/borneolcc/lc1960/{rver}/a-A-2019-09-16-140000-g1.h5",
    drop_variables=[
        v for v in ana_var if v not in ["LEAF_CLASS", "PATCH_AREA"]
    ],
    engine="h5netcdf",
    chunks="auto",
    phony_dims="access",
).rename_dims({"phony_dim_0": "y", "phony_dim_1": "x", "phony_dim_3": "p"})

coord["lc"] = (coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim=("p"))
landmask = coord.lc != 0


def get_land(ds, landmask):
    return ds * landmask


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})

variables = ["SFLUX_T", "SFLUX_R", "LWUP", "LWDN", "SWUP", "SWDN"]
drop_var = [v for v in lite_var if v not in variables]
print(drop_var)

for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 24)):
    times = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]
    savePath = f"/squall/gleung/borneolcc-analysis/seb_land_{rver}_{str(i).zfill(2)}.nc"

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
    ds = ds.unify_chunks()

    out = xr.Dataset()

    out = out.assign(lhf=ds.SFLUX_R * lv)
    out = out.assign(shf=ds.SFLUX_T * cp)

    out = out.assign(lwdn=ds.LWDN.sel(z=1))
    out = out.assign(lwup=ds.LWUP.sel(z=1))
    out = out.assign(swdn=ds.SWDN.sel(z=1))
    out = out.assign(swup=ds.SWUP.sel(z=1))

    land = out * landmask

    land = land.sum(dim=("x", "y")) / landmask.sum(dim=("x", "y"))

    land = land.compute()

    land.to_netcdf(savePath, engine="h5netcdf", mode="w")
