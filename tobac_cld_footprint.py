import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import labeled_comprehension, find_objects
import dask
import dask.distributed as dd

client = dd.Client("snowfall2:8786")
client.upload_file("shared_model_params.py")


from shared_model_params import (
    get_rams_output,
    compute_cond,
    compute_pcp,
    alt,
    dz,
)


dxy = 150


def calculate_footprint(ft, loc, arr):
    mask = arr.sel(z=loc[0], y=loc[1], x=loc[2]) == ft
    return get_footprint(mask.data).compute()


def get_footprint(arr):
    return arr.max(axis=(0)).sum()


n = 24

for lc in ["lc2019"]:
    dataPath = f"/squall/gleung/borneolcc/{lc}/rte/"
    tobacPath = f"/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/"
    figPath = f"/squall/gleung/borneolcc-figures/tobac-testing/{lc}-rte/"

    if not os.path.isdir(figPath):
        os.mkdir(figPath)

    tracks = pd.read_parquet(f"{tobacPath}/final_cond-w_segmented_tracks.pq")

    out_df = []

    for frame in sorted(tracks.frame.unique()):
        sub = tracks[tracks.frame == frame]
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        cond_mask = xr.open_dataset(
            f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        locs = find_objects(cond_mask.segmentation_mask)
        locs = [l for l in locs if l != None]
        locs = pd.DataFrame(locs, index=fts[sub["condensate_volume"] > 0])
        locs = [x[1] for x in locs.iterrows()]

        x = client.map(
            calculate_footprint,
            fts[sub["condensate_volume"] > 0],
            locs,
            arr=cond_mask.segmentation_mask,
        )
        x = client.gather(x)
        x = pd.DataFrame(x, index=fts[sub["condensate_volume"] > 0])
        if len(x > 0):
            sub["cond_footprint"] = sub.feature.map(x[0]) * (
                dxy * dxy / (1000**2)
            )

            out_df.append(sub)
        else:
            print(frame, sub)

    out_df = pd.concat(out_df)

    out_df.to_parquet(f"{tobacPath}/cloud_footprint_statistics.pq")
