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

client = dd.Client("tcp://10.1.20.22:8786")
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
rver = "har"

dataPath = f"/squall/gleung/borneolcc/{lcver}/{rver}/"
grid = "g1"
figPath = f"/squall/gleung/borneolcc-figures/{lcver}/{rver}/quicklook/"

if not os.path.isdir(figPath):
    os.mkdir(figPath)

all_paths = [p for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))]
print(len(all_paths))

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

def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})

variables = [
    "FTHRDSW",
    "FTHRDLW",
    "FTHRD",
    "LWUP",
    "LWDN",
    "SWUP",
    "SWDN",
    "THETA",
    "RV",
    "PI",
    "RCP",
    "RPP",
    "RSP",
    "RDP",
    "RRP",
    "RAP",
    "RGP",
    "RHP",
]

drop_var = [v for v in lite_var if v not in variables]
]

# dask seems to work better when you break up into shorter time intervals (i.e., number of files it needs to open)
# I think this is mainly just to prevent running into memory limits
# here I am splitting up my full list of times into groups of 24 (which for my 5 min output means 5 hrs at a time)
for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 24)):
    times = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]

    # defining some save paths
    saveallPath = f"/squall/gleung/borneolcc-analysis/radprof/radprof_all_{lcver}_{rver}_{str(i).zfill(2)}.nc"
    saveclrPath = f"/squall/gleung/borneolcc-analysis/radprof/radprof_clr_{lcver}_{rver}_{str(i).zfill(2)}.nc"
    savecldPath = f"/squall/gleung/borneolcc-analysis/radprof/radprof_cld_{lcver}_{rver}_{str(i).zfill(2)}.nc"

    # just check to make sure we aren't re-doing calculations that are already finished
    if not os.path.exists(saveallPath):
        # read in files in paths (subset of all_paths)
        ds = xr.open_mfdataset(
            paths,
            engine="h5netcdf",
            chunks="auto",
            phony_dims="access",
            drop_variables=drop_var,  # use this to drop variables before reading them in, helps with memory rather than reading everything in then dropping variables later
            combine="nested",
            concat_dim=[pd.Index(times, name="t")],
            parallel=True,  # using this setting seems to speed up dask computations, not sure if I'm imagining that
        )

        ds = rename_dims(ds, rams_dims_lite)
        ds = (
            ds.unify_chunks()
        )  # also seems to help with speed;makes sure all chunks are uniform across variables

        # creating some variables for cloud/precipitating ice/liquid
        ds = ds.assign(
            CICE=(
                ("time", "z", "y", "x"),
                dask.array.map_blocks(
                    lambda x, y: x + y,
                    ds.RPP.data,
                    ds.RSP.data,
                    dtype=np.float64,
                ),
            )
        )

        ds = ds.assign(CLIQ=ds.RCP)

        ds = ds.assign(
            PLIQ=(
                ("time", "z", "y", "x"),
                dask.array.map_blocks(
                    lambda x, y: x + y,
                    ds.RDP.data,
                    ds.RRP.data,
                    dtype=np.float64,
                ),
            )
        )

        ds = ds.assign(
            PICE=(
                ("time", "z", "y", "x"),
                dask.array.map_blocks(
                    lambda x, y, z: x + y + z,
                    ds.RAP.data,
                    ds.RHP.data,
                    ds.RGP.data,
                    dtype=np.float64,
                ),
            )
        )

        # I don't need every single hydrometeor variable, so get rid of them
        ds = ds[
            [
                "CLIQ",
                "CICE",
                "PLIQ",
                "PICE",
                "FTHRDSW",
                "FTHRDLW",
                "FTHRD",
                "LWUP",
                "LWDN",
                "SWUP",
                "SWDN",
                "THETA",
                "RV",
                "PI",
            ]
        ]

        # remove all the gridpoints which are water, not interested in them here
        land = ds * landmask
        # from the land cells, define a cloud mask for a given mixing ratio threshold (I use 0.001 g/kg)
        cldmask = (land.CLIQ + land.CICE) >= 1.0e-5

        # takes all land points which are also cloudy
        cld = land * cldmask
        cld = cld.sum(dim=("x", "y")) / cldmask.sum(dim=("x", "y"))
        # make sure to compute before saving, if I don't do this I sometimes end up with files that are just nan's
        cld = cld.compute()
        cld.to_netcdf(savecldPath, engine="h5netcdf", mode="w")

        # land points which are not cloudy
        clr = land * ~cldmask
        clr = clr.sum(dim=("x", "y")) / (~cldmask).sum(dim=("x", "y"))
        clr = clr.compute()
        clr.to_netcdf(savecldPath, engine="h5netcdf", mode="w")

        land = land.sum(dim=("x", "y")) / landmask.sum(dim=("x", "y"))
        land = land.compute()
        land.to_netcdf(saveallPath, engine="h5netcdf", mode="w")
