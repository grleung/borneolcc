"""
Step 1 of tobac processing: Identify updraft features

Input: RAMS lite files
Output: parquet file with features 'w_features.pq'
"""

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
client = dd.Client("snowfall2:8786")
client.upload_file("shared_model_params.py")

from shared_model_params import get_rams_output, save_files, dx

modelPath = "/squall/gleung/borneolcc/"
outPath = f"/squall/gleung/borneolcc-analysis/tobac/"

# tobac feature identification parameters
# see tobac documentation for more detailed description
params = {}
params["position_threshold"] = "weighted_diff"
params["sigma_threshold"] = 1
params["n_erosion_threshold"] = 0
# this is ~4 points in each direction; we had some discussion about changing
# this for different grid spacings, but as of 2024-08-13 this makes more sense to me (Bee)
params["n_min_threshold"] = 64
params["target"] = "maximum"
# threshold in m/s is (1, 2, 4, 6, ..., 50)
params["threshold"] = np.append([1.0], np.arange(2.0, 52.0, 2.0))
dxy = dx

# loop over two simulations
for lc in ["lc1960", "lc2019"]:
    dataPath = f"{modelPath}/{lc}/rte/"

    # list of all timesteps where lite files are found in relevant folder
    all_paths = [
        p.split("/")[-1][:-6] for p in sorted(glob.glob(f"{dataPath}/a-L-*-g1.h5"))
    ]
    all_paths = all_paths[
        6 * 12 : ((6 * 12) + (3 * 24 * 12)) + 1
    ]  # first 6 hours are spinup; analysis period is first 3 days after spinup period

    # make sure path exists
    if not os.path.exists(f"{outPath}/{lc}_rte/"):
        os.mkdir(f"{outPath}/{lc}_rte/")

    # split the paths into smaller groups so everything fits into memory
    for i, paths in enumerate(np.array_split(all_paths, len(all_paths) // 24)):
        print(paths)

        savedfPath = f"{outPath}/{lc}_rte/w_features_{str(i).zfill(2)}.pq"

        if not os.path.exists(savedfPath):
            try:
                # prep data for feeding to tobac
                ds = client.map(
                    get_rams_output,
                    [f"{dataPath}/{p}-g1.h5" for p in paths],
                    variables=["WP"],
                    prep_tobac=True,
                )

                # time needs to be a dimension
                ds = client.map(
                    xr.DataArray.expand_dims,
                    ds,
                    [{"time": [pd.to_datetime(p.split("/")[-1][4:])]} for p in paths],
                )

                # actual tobac run
                feats = client.map(
                    tobac.feature_detection_multithreshold,
                    ds,
                    dxy=dxy,
                    vertical_coord="ztn",
                    **params,
                )

                # take all the features from tobac run
                all_features = client.gather(feats)

                # once loop is finished, concatenate all the figures
                # then save it to a parquet file
                all_features = tobac.utils.combine_feature_dataframes(all_features)

                save_files(all_features, savedfPath)

            except TimeoutError:
                pass

    savePaths = sorted(glob.glob(f"{outPath}/{lc}_rte/w_features_*.pq"))

    # make sure all the saved files are present
    if len(savePaths) == (len(all_paths) // 24):
        # read in all the files, combine, and save

        all_df = pd.read_parquet(savePaths, engine="pyarrow")

        # use tobac tool so that frame numbering is correct
        all_df = tobac.utils.combine_feature_dataframes([all_df])

        # there shouldn't be any duplicates, but just in case
        all_df = all_df.drop_duplicates(["frame", "x", "y", "z"])

        all_df.to_parquet(f"{outPath}/{lc}_rte/w_features.pq")

        print(len(all_df.time.unique()))
