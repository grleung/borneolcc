import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import labeled_comprehension, find_objects
import dask
import dask.distributed as dd

client = dd.Client("anvil:9999")
client.upload_file("shared_model_params.py")


dxy = 150


def calculate_footprint(ft, loc, arr):
    mask = arr.sel(z=loc[0], y=loc[1], x=loc[2]) == ft
    return get_footprint(mask.data).compute()


def get_footprint(arr):
    return arr.max(axis=(0)).sum()


for lc in ["lc2019"]:
    dataPath = f"/squall/gleung/borneolcc/{lc}/rte/"
    tobacPath = f"/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/"

    tracks = pd.read_parquet(f"{tobacPath}/cloudy_updraft_statistics.pq")
    tracks = tracks[tracks.cond_ncells > 0]

    if not os.path.exists(f"{tobacPath}/cloudy_updraft_statistics_full.pq"):

        out_df = []

        for frame in sorted(tracks.frame.unique()):
            print(frame)
            sub = tracks[tracks.frame == frame]
            fts = sub.feature.unique()
            time = pd.to_datetime(sub.timestr.iloc[0])

            cond_mask = xr.open_dataset(
                f"{tobacPath}/cond_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
                engine="h5netcdf",
                chunks="auto",
            )

            locs = find_objects(cond_mask.segmentation_mask)
            locs = [
                l for i, l in enumerate(locs) if (l != None) and (i + 1 in fts)
            ]

            locs = pd.DataFrame(locs, index=fts[sub["condensate_volume"] > 0])
            locs = locs.loc[fts]
            locs = [x[1] for x in locs.iterrows()]

            print(len(locs), len(fts))

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

        out_df.to_parquet(f"{tobacPath}/cloudy_updraft_statistics_full.pq")
