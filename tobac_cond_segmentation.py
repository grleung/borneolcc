# Import some shared libraries
import os
import dask.distributed as dd
import dask
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob

# change this address depending on your scheduler address
client = dd.Client("downdraft:8786")
client.upload_file("shared_model_params.py")

from shared_model_params import (
    get_rams_output,
    combine_tobac_list,
    save_files,
    compute_cond,
)


modelPath = "/squall/gleung/borneolcc/"
outPath = f"/squall/gleung/borneolcc-analysis/tobac/"

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 1.e-5 #1.0e-5  # kg/m3 mixing ratio
params["seed_3D_flag"] = "box"
params["vertical_coord"] = "ztn"

#doing a test with box size for seeding
params["seed_3D_size"] = (25,5,5)


aero = "rte"

for lc in ['lc1960']:
    dataPath = f"{modelPath}/{lc}/{aero}/"
    # list of all timesteps where lite files are found in relevant folder

    all_paths = [
        p.split("/")[-1][:-6]
        for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))
    ]

    if lc=='lc1960':
        all_paths = all_paths[(12*6)+(12*24*3):]#all_paths[(12*6):(12*6)+(12*24*3)]
    elif lc=='lc2019':
        all_paths = all_paths[(12*6)+(12*24*3)+1:]#all_paths[(12*6):(12*6)+(12*24*3)+1]

    print(all_paths)
    
    all_times = [pd.to_datetime(p.split("/")[-1][4:]) for p in all_paths]

    dxy = 150

    trackPath = f"{outPath}/{lc}_{aero}/w_tracks-new.pq"
    tracks = pd.read_parquet(trackPath)

    all_paths = [
        p
        for i, p in enumerate(all_paths)
        if (all_times[i] in tracks.time.unique())
    ]

    print(lc, len(all_paths))

    savemaskPath = f"{outPath}/{lc}_{aero}/cond_masks_column_anvil/"
    if not os.path.isdir(savemaskPath):
        os.mkdir(savemaskPath)

    for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 12)):
        i = i+68
        print(lc, i)
        savedfPath = (
            f"{outPath}/{lc}_{aero}/cond_column_anvil_segmentation_{str(i).zfill(2)}.pq"
        )

        if (i==95) and (not os.path.exists(savedfPath)):
            print(i)
            times = [pd.to_datetime(p.split("/")[-1][4:]) for p in paths]

            # prep data for feeding to tobac
            ds = client.map(
                get_rams_output,
                [f"{dataPath}/{p}-g1.h5" for p in paths],
                variables=["RCP", "RSP", "RPP", "PI", "THETA", "RV"],
                prep_tobac=True,
            )

            ds = client.map(compute_cond, ds)

            ds = client.map(
                xr.DataArray.expand_dims,
                ds,
                [
                    {"time": [pd.to_datetime(p.split("/")[-1][4:])]}
                    for p in paths
                ],
            )

            ds = client.map(
                xr.DataArray.to_iris,
                ds,
            )

            out = client.map(
                tobac.segmentation.segmentation,
                [tracks[tracks.time == t] for t in times],
                ds,
                dxy=dxy,
                **params,
            )

            out = client.gather(out)

            all_segments = [o[1] for o in out]
            all_masks = [o[0] for o in out]

            # once loop is finished, concatenate all the figures
            # then save it to a parquet file
            all_segments = combine_tobac_list(all_segments)
            save_files(all_segments, savedfPath)

            for m, p in zip(all_masks, paths):
                ds = xr.DataArray.from_iris(m)
                ds.to_netcdf(
                    f"{savemaskPath}/{p}.h5",
                    engine="h5netcdf",
                    encoding={
                        "segmentation_mask": {"zlib": True, "complevel": 9}
                    },
                )
