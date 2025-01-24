import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import dask
import dask.distributed as dd
import glob

client = dd.Client("snowfall1:8786")
client.upload_file("shared_model_params.py")
from shared_model_params import (
    get_rams_output,
    rams_dims_lite,
    rams_dims_anal,
    lite_var,
)

# Soil parameters
soil_sat = np.array(
    [
        0.395,
        0.410,
        0.435,
        0.485,
        0.451,
        0.420,
        0.477,
        0.476,
        0.426,
        0.492,
        0.482,
        0.863,
    ]
)


soil_heatcap = np.array(
    [
        1465e3,
        1407e3,
        1344e3,
        1273e3,
        1214e3,
        1177e3,
        1319e3,
        1227e3,
        1177e3,
        1151e3,
        1088e3,
        874e3,
    ]
)


def get_soil_sat(i):
    return soil_sat[i - 1]


def get_soil_heatcap(i):
    return soil_heatcap[i - 1]


def assign_soil_sat(ds):
    # assigns soil saturation water content based on soil type
    ds = ds.assign(
        SOIL_SATWAT=(
            ("y", "x"),  # "g"),
            get_soil_sat(ds.SOIL_TEXT.values.astype(int)),
        )
    )

    return ds


def assign_soil_heatcap(ds):
    # assigns soil dry heat capacity based on soil type
    ds = ds.assign(
        SOIL_HEATCAP=(
            ("y", "x"),  # , "g"),
            get_soil_heatcap(ds.SOIL_TEXT.values.astype(int)),
        )
    )

    return ds


def calculate_soil_temp(soil_energy, soil_water, soil_heatcap):
    # calculate soil temp in degC (from REVU)
    if soil_energy <= 0:
        return soil_energy / ((2.093e6 * soil_water) + soil_heatcap)
    elif soil_energy >= soil_water * 3.34e8:
        return (soil_energy - (soil_water * 3.34e8)) / (
            (4.186e6 * soil_water) + soil_heatcap
        )
    else:
        return 0


calculate_soil_temp = np.vectorize(calculate_soil_temp)


def assign_soil_temp(ds):
    # assigns soil temperature
    ds = ds.assign(
        SOIL_TEMP=(
            ("y", "x",'time'),  # "g"),
            calculate_soil_temp(ds.SOIL_ENERGY, ds.SOIL_WATER, ds.SOIL_HEATCAP.expand_dims(time=ds.time,axis=-1)),
        )
    )

    return ds


lcver = "lc1960"
dataPath = f"/squall/gleung/borneolcc/{lcver}/rte/"

all_paths = [p for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))]

# these 2d things are fixed in time
coord = get_rams_output(f"/squall/gleung/borneolcc/{lcver}/rte/a-A-2019-09-16-140000-g1.h5",
    variables=[
        "SOIL_TEXT",
        "PATCH_AREA",
        "LEAF_CLASS",
    ],
    dims=rams_dims_anal)[[
        "SOIL_TEXT",
        "PATCH_AREA",
        "LEAF_CLASS",
    ]]

coord = coord.sel(g=10)
coord = coord.assign(SOIL_TEXT=(coord.PATCH_AREA * coord.SOIL_TEXT).sum(dim="p"))
coord = coord.assign(LCTYPE=(coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim="p"))

coord["lc"] = (coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim=("p"))
landmask = coord.lc != 0


def get_land(ds, landmask):
    return ds * landmask


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


rams_dims_lite.update({"t": "time"})

variables = ['SOIL_WATER','SOIL_ENERGY']
drop_var = [v for v in lite_var if v not in variables]
print(drop_var)

print(len(np.array_split(all_paths, len(all_paths) // 36)))

for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 36)):
    times = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]
    savePath = f"/squall/gleung/borneolcc-analysis/surf/soil_land_{lcver}_rte_{str(i).zfill(2)}.nc"

    if  (not os.path.exists(savePath)):
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
	    ds = ds.sel(g=10)

	    out = xr.Dataset()

	    out = out.assign(SOIL_WATER = (coord.PATCH_AREA*ds.SOIL_WATER).sum(dim='p'))
	    out = out.assign(SOIL_ENERGY = (coord.PATCH_AREA*ds.SOIL_ENERGY).sum(dim='p'))

	    out = out.assign({'SOIL_TEXT':coord.SOIL_TEXT,"LCTYPE":coord.LCTYPE})

	    out = assign_soil_sat(out)
	    out = assign_soil_heatcap(out)
	    out = assign_soil_temp(out)
	    out = out.assign(SOIL_SATFRAC = out.SOIL_WATER/out.SOIL_SATWAT)
	    
	    land = out * landmask

	    land = land.sum(dim=("x", "y")) / landmask.sum(dim=("x", "y"))
	    land = land[['SOIL_ENERGY','SOIL_TEMP','SOIL_WATER','SOIL_SATFRAC']]

	    land = land.compute()

	    land.to_netcdf(savePath, engine="h5netcdf", mode="w")
