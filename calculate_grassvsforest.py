import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import dask
import dask.distributed as dd
import glob

client = dd.Client("snowfall2:8786")
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
all_paths = [p for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))][
    ((24 * 3) + 6) * 12 :
]


# these 2d things are fixed in time
coord = xr.open_dataset(
    f"/squall/gleung/borneolcc/lc2019/{rver}/a-A-2019-09-16-140000-g1.h5",
    drop_variables=[
        v for v in ana_var if v not in ["LEAF_CLASS", "PATCH_AREA"]
    ],
    engine="h5netcdf",
    chunks="auto",
    phony_dims="access",
).rename_dims({"phony_dim_0": "y", "phony_dim_1": "x", "phony_dim_3": "p"})

coord["lc"] = (coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim=("p"))

grassmask = coord.lc == 18
forestmask = coord.lc == 7


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})

variables = [
    "SFLUX_T",
    "SFLUX_R",
    "LWUP",
    "LWDN",
    "SWUP",
    "SWDN",
    "PCPRR",
    "PCPRD",
    "PCPRA",
    "PCPRS",
    "PCPRH",
    "PCPRG",
    "PCPRP",
]
drop_var = [v for v in lite_var if v not in variables]
print(drop_var)

print(len(np.array_split(all_paths, len(all_paths) // 72)))


for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 72)):
    i = i + 13
    times = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]
    saveforestPath = f"/squall/gleung/borneolcc-analysis/seb/seb_forest_{lcver}_{rver}_{str(i).zfill(2)}.nc"
    savegrassPath = f"/squall/gleung/borneolcc-analysis/seb/seb_grass_{lcver}_{rver}_{str(i).zfill(2)}.nc"

    if not os.path.exists(saveforestPath) and (
        not os.path.exists(savegrassPath)
    ):
        print(i)

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

        out = out.assign(
            pcp=(
                ds.PCPRR
                + ds.PCPRD
                + ds.PCPRA
                + ds.PCPRS
                + ds.PCPRH
                + ds.PCPRG
                + ds.PCPRP
            )
            * 3600
        )

        out = out.assign(lhf=ds.SFLUX_R * lv)
        out = out.assign(shf=ds.SFLUX_T * cp)

        out = out.assign(lwdn=ds.LWDN.sel(z=1))
        out = out.assign(lwup=ds.LWUP.sel(z=1))
        out = out.assign(swdn=ds.SWDN.sel(z=1))
        out = out.assign(swup=ds.SWUP.sel(z=1))

        grass = out * grassmask
        grass = grass.sum(dim=("x", "y")) / grassmask.sum(dim=("x", "y"))
        grass = grass.compute()
        grass.to_netcdf(savegrassPath, engine="h5netcdf", mode="w")

        forest = out * forestmask
        forest = forest.sum(dim=("x", "y")) / forestmask.sum(dim=("x", "y"))
        forest = forest.compute()
        forest.to_netcdf(saveforestPath, engine="h5netcdf", mode="w")
