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
client = dd.Client("solvarg:8786")
client.upload_file("shared_model_params.py")

from shared_model_params import get_rams_output, save_files

client.upload_file("shared_processing.py")
from shared_processing import (
    compute_pcp,
)


modelPath = "/squall/gleung/borneolcc/"
outPath = f"/squall/gleung/borneolcc-analysis/tobac/"

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 0.01  # mm/hr mixing ratio

for lc in ["lc2019", "lc1960"]:
    print(lc)

    dataPath = f"{modelPath}/{lc}/rte/"
    # list of all timesteps where lite files are found in relevant folder

    all_paths = [
        p.split("/")[-1][:-6]
        for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))
    ]
    all_times = [pd.to_datetime(p.split("/")[-1][4:]) for p in all_paths]

    dxy = 150

    trackPath = f"{outPath}/{lc}_rte/cloudy_updrafts.pq"  # combined_cond-w_segmented_tracks.pq"
    tracks = pd.read_parquet(trackPath)

    paths = [
        p
        for i, p in enumerate(all_paths)
        if (all_times[i] in tracks.time.values)
    ]

    print(len(paths))
    print(paths)

    savemaskPath = f"{outPath}/{lc}_rte/pcp_masks/"

    if not os.path.isdir(savemaskPath):
        os.mkdir(savemaskPath)

    savedfPath = f"{outPath}/{lc}_rte/cloudy_updrafts_raining.pq"  # combined_cond-w-pcp_segmented_tracks.pq"

    if not os.path.exists(savedfPath):
        times = [pd.to_datetime(p.split("/")[-1][4:]) for p in paths]

        # prep data for feeding to tobac
        ds = client.map(
            get_rams_output,
            [f"{dataPath}/{p}-g1.h5" for p in paths],
            variables=[
                "PCPRR",
                "PCPRP",
                "PCPRS",
                "PCPRA",
                "PCPRG",
                "PCPRH",
                "PCPRD",
            ],
            prep_tobac=True,
        )

        ds = client.map(compute_pcp, ds)

        ds = client.map(
            xr.DataArray.expand_dims,
            ds,
            [{"time": [t]} for t in times],
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
        pcp = pd.concat(all_segments)

        tracks = tracks.set_index(["time", "cell"])
        pcp = pcp.set_index(["time", "cell"])

        print(pcp.columns)

        tracks["pcp_ncells"] = tracks.index.map(pcp.ncells)
        tracks = tracks.reset_index()
        tracks["cellmax_pcpncells"] = tracks.groupby(
            "cell"
        ).pcp_ncells.transform("max")

        save_files(tracks, savedfPath)

        for m, p in zip(all_masks, paths):
            m.to_netcdf(
                f"{savemaskPath}/{p}.h5",
                engine="h5netcdf",
                encoding={"segmentation_mask": {"zlib": True, "complevel": 9}},
            )
