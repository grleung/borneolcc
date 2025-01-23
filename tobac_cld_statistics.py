import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import (
    labeled_comprehension,
    maximum_position,
    sum_labels,
    maximum,
    mean,
    minimum,
)
import dask
import dask.distributed as dd

client = dd.Client("snowfall1:8786")
client.upload_file("shared_model_params.py")


from shared_model_params import (
    get_rams_output,
    compute_cond,
    compute_pcp,
    alt,
    dz,
)

dxy = 150


def get_masked_statistics(sub, dataPath, tobacPath):
    if len(sub) > 0:
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        cond_mask = xr.open_dataset(
            f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        cond_mask = cond_mask.assign(dz=("z", dz))

        shape = (
            cond_mask.segmentation_mask / cond_mask.segmentation_mask
        ).fillna(1)
        cond_dz = cond_mask.dz * shape
        cond_alt = ((cond_mask.ztn) / 1000) * shape

        sub["CTH"] = labeled_comprehension(
            cond_alt / 1000,
            cond_mask.segmentation_mask,
            fts,
            np.nanmax,
            np.float64,
            np.nan,
        )

        sub["CBH"] = labeled_comprehension(
            cond_alt / 1000,
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

for lc in ["lc1960"]:
    dataPath = f"/squall/gleung/borneolcc/{lc}/rte/"
    tobacPath = f"/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/"
    figPath = f"/squall/gleung/borneolcc-figures/tobac-testing/{lc}-rte/"

    if not os.path.isdir(figPath):
        os.mkdir(figPath)

    tracks = pd.read_parquet(f"{tobacPath}/combined_cond-w_segmented_tracks.pq")

    frames = tracks.frame.unique()
    times = tracks.time.unique()
    if lc == "lc2019":
        frames = frames[times >= pd.to_datetime("2019-09-19 19:00")]
    elif lc == "lc1960":
        frames = frames[times >= pd.to_datetime("2019-09-18 18:00")]

    for i, frames in enumerate(
        np.array_split(sorted(frames), len(frames) // n)
    ):
        if lc == "lc2019":
            i = i + 28
        elif lc == "lc1960":
            i = i + 29

        print(lc, i)

        if not os.path.exists(
            f"{tobacPath}/cloud_statistics_{str(i).zfill(2)}.pq"
        ):
            x = client.map(
                get_masked_statistics,
                [tracks[tracks.frame == frame] for frame in frames],
                dataPath=dataPath,
                tobacPath=tobacPath,
            )
            x = client.gather(x)

            x = pd.concat(x)

            x.to_parquet(f"{tobacPath}/cloud_statistics_{str(i).zfill(2)}.pq")
