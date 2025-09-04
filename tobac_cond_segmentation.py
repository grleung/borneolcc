"""
Step 4 of tobac processing: Segment condensate features

Input: parquet file with tracked features 'w_tracks.pq' + RAMS lite files
Output: parquet file with segmented features 'cond_segmentation.pq' + condensate mask files per timestep
"""

# Import some shared libraries
import os
import dask.distributed as dd
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob

# change this address depending on your scheduler address
client = dd.Client("solvarg:8786")
client.upload_file("shared_model_params.py")
client.upload_file("shared_processing.py")
from shared_model_params import get_rams_output, save_files, dx
from shared_processing import compute_cond

dxy = dx

modelPath = "/squall/gleung/borneolcc/"
outPath = f"/squall/gleung/borneolcc-analysis/tobac/"

# parameters for segmentation
params = {}
params["method"] = "watershed"
params["threshold"] = 1.0e-5  # 1.0e-5  # kg/m3 mixing ratio
params["seed_3D_flag"] = "box"
params["vertical_coord"] = "ztn"
# set the vertical box size to be 25 so it doesn't miss clouds where w centroid is below the LCL
params["seed_3D_size"] = (25, 5, 5)


def dask_cond_segmentation(path, lc):
    dataPath = f"{modelPath}/{lc}/rte/"
    savemaskPath = f"{outPath}/{lc}_rte/cond_masks/"

    time = pd.to_datetime(path.split("/")[-1][4:])

    ds = get_rams_output(
        f"{dataPath}/{path}-g1.h5",
        variables=["RCP", "RSP", "RPP", "PI", "THETA", "RV"],
        prep_tobac=True,
    )
    ds = compute_cond(ds)
    ds = ds.expand_dims({"time": [time]})

    tracks = pd.read_parquet(f"{outPath}/{lc}_rte/w_tracks.pq")
    tracks = tracks[tracks.time == time].reset_index(drop=True)

    if len(tracks) > 0:

        mask, seg = tobac.segmentation.segmentation(
            tracks,
            ds,
            dxy=dxy,
            **params,
        )

        mask.to_netcdf(
            f"{savemaskPath}/{path}.h5",
            engine="h5netcdf",
            encoding={"segmentation_mask": {"zlib": True, "complevel": 9}},
        )

        del ds
        del mask

        return seg
    else:
        return pd.DataFrame(
            columns=[
                "frame",
                "idx",
                "vdim",
                "hdim_1",
                "hdim_2",
                "num",
                "threshold_value",
                "feature",
                "time",
                "timestr",
                "z",
                "y",
                "x",
                "ztn",
                "lat",
                "lon",
                "cell",
                "time_cell",
                "lifetime",
                "ncells",
            ]
        )


for lc in ["lc1960"]:
    dataPath = f"{modelPath}/{lc}/rte/"

    # list of all timesteps where lite files are found in relevant folder
    all_paths = [
        p.split("/")[-1][:-6] for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))
    ]
    all_paths = all_paths[
        6 * 12 : ((6 * 12) + (3 * 24 * 12)) + 1
    ]  # first 6 hours are spinup; analysis period is first 3 days after spinup period

    all_times = [pd.to_datetime(p.split("/")[-1][4:]) for p in all_paths]

    savemaskPath = f"{outPath}/{lc}_rte/cond_masks/"
    if not os.path.isdir(savemaskPath):
        os.mkdir(savemaskPath)

    # split into smaller groups to fit into memory
    for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 12)):
        print(lc, i)
        savedfPath = f"{outPath}/{lc}_rte/cond_segmentation_{str(i).zfill(2)}.pq"
        print(savedfPath)

        if not os.path.exists(savedfPath):
            print(i)

            # call segmentation
            out = client.map(dask_cond_segmentation, paths, lc=lc)
            out = client.gather(out)

            # once loop is finished, concatenate all the figures
            # then save it to a parquet file
            all_segments = pd.concat(out)

            save_files(all_segments, savedfPath)

    savePaths = sorted(glob.glob(f"{outPath}/{lc}_rte/cond_segmentation_*.pq"))

    # make sure all the saved files are present
    if len(savePaths) == (len(all_paths) // 12):
        # read in all the files, combine, and save

        tracks = pd.read_parquet(f"{outPath}/{lc}_rte/w_tracks.pq")
        tracks = tracks.drop_duplicates(["timestr", "hdim_1", "hdim_2"])
        tracks = tracks.set_index(["timestr", "hdim_1", "hdim_2"]).sort_index()

        all_df = []
        for p in savePaths:
            all_df.append(pd.read_parquet(p, engine="pyarrow"))
        all_df = pd.concat(all_df)

        all_df["ncells"] = all_df.ncells.fillna(0)

        all_df = all_df.drop_duplicates(["timestr", "hdim_1", "hdim_2"])
        all_df = all_df.set_index(["timestr", "hdim_1", "hdim_2"]).sort_index()

        all_df["frame"] = all_df.index.map(tracks.frame)
        all_df["feature"] = all_df.index.map(tracks.feature)
        all_df = all_df.reset_index().dropna(subset=["frame", "feature"])
        all_df.to_parquet(f"{outPath}/{lc}_rte/cond_segmentation.pq")

        print(len(all_df.time.unique()))
