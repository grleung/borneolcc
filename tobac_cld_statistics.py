"""
Step 7 of tobac processing: Calculate cloud statistics

Input: parquet file with merged features 'cloudy_updrafts.pq' + condensate masks
Output: parquet file with features 'cloudy_updraft_statistics.pq'
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import (
    labeled_comprehension,
    sum_labels,
)
import dask.distributed as dd
import glob

client = dd.Client("solvarg:8786")
client.upload_file("shared_model_params.py")


from shared_model_params import (
    dz,
)

dxy = 150


def get_masked_statistics(frame, tobacPath):
    tracks = pd.read_parquet(f"{tobacPath}/cloudy_updrafts.pq")
    tracks = tracks[tracks.cond_ncells > 0]
    sub = tracks[tracks.frame == frame]

    if len(sub) > 0:
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        cond_mask = xr.open_dataset(
            f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        cond_mask = cond_mask.assign(dz=("z", dz))

        shape = (cond_mask.segmentation_mask / cond_mask.segmentation_mask).fillna(1)
        cond_dz = cond_mask.dz * shape
        cond_alt = ((cond_mask.ztn) / 1000) * shape

        sub["CTH"] = labeled_comprehension(
            cond_alt,
            cond_mask.segmentation_mask,
            fts,
            np.nanmax,
            np.float64,
            np.nan,
        )

        sub["CBH"] = labeled_comprehension(
            cond_alt,
            cond_mask.segmentation_mask,
            fts,
            np.nanmin,
            np.float64,
            np.nan,
        )

        sub["condensate_volume"] = (
            sum_labels(
                cond_dz,
                cond_mask.segmentation_mask,
                fts,
            )
            * dxy
            * dxy
            / (1000 * 1000 * 1000)
        )

        sub["condensate_count"] = sum_labels(
            shape,
            cond_mask.segmentation_mask,
            fts,
        )

        print("cloud stats")

        return sub
    else:
        print(sub)


n = 24

for lc in ["lc1960", "lc2019"]:
    dataPath = f"/squall/gleung/borneolcc/{lc}/rte/"
    tobacPath = f"/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/"

    tracks = pd.read_parquet(f"{tobacPath}/cloudy_updrafts.pq")
    tracks = tracks[tracks.cond_ncells > 0]

    frames = tracks.frame.unique()
    times = tracks.time.unique()

    print(len(frames) // n)
    for i, frames_ in enumerate(np.array_split(sorted(frames), len(frames) // n)):
        print(lc, i)

        if not os.path.exists(
            f"{tobacPath}/cloudy_updraft_statistics_{str(i).zfill(2)}.pq"
        ):
            x = client.map(
                get_masked_statistics,
                frames_,
                tobacPath=tobacPath,
            )
            x = client.gather(x)

            x = pd.concat(x)

            x.to_parquet(f"{tobacPath}/cloudy_updraft_statistics_{str(i).zfill(2)}.pq")

    savePaths = sorted(glob.glob(f"{tobacPath}/cloudy_updraft_statistics_*.pq"))
    print(len(savePaths))
    print(len(frames) // 24)

    # make sure all the saved files are present
    if len(savePaths) == (len(frames) // 24):
        # read in all the files, combine, and save

        all_df = []
        for p in savePaths:
            all_df.append(pd.read_parquet(p, engine="pyarrow"))
        all_df = pd.concat(all_df)

        all_df.to_parquet(f"{tobacPath}/cloudy_updraft_statistics.pq")

        print(len(all_df.time.unique()))
