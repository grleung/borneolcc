# Import some shared libraries
import os
import dask.distributed as dd
import dask
import numpy as np
import pandas as pd
import xarray as xr
import tobac
import glob
import datetime as dt
from shared_model_params import get_rams_output, dz, combine_tobac_list

outPath = "/squall/gleung/borneolcc-analysis/tobac/"
lcs = ["lc1960", "lc2019"]

for lc in lcs:
    print(lc)

    paths = sorted(glob.glob(f"{outPath}/{lc}_rte/w_features_*.pq"))
    print(paths)

    all_df = []
    for p in paths:
        df = pd.read_parquet(p, engine="pyarrow")
        df = df[df.time >= pd.to_datetime("2019-09-16-20")]

        all_df.append(df)
        print(p, len(df))

    # use tobac tool so that frame numbering is correct
    all_df = combine_tobac_list(all_df)

    all_df.to_parquet(f"{outPath}/{lc}_rte/w_features-new.pq")

# tobac tracking parameters
# see tobac documentation for more detailed description of each parameter
params = {}
params["extrapolate"] = 0
params["order"] = 1
params["memory"] = 0
params["time_cell_min"] = 15 * 60  # in seconds
params["method_linking"] = "predict"
params["adaptive_step"] = 0.75
params["adaptive_stop"] = 1.0
params["d_max"] = 150 * 10

n = 50  # number of points from boundaries to exclude, there are 25 pts which are being nudged so this is ~7.5km

for lc in lcs:
    print(lc)
    features = pd.read_parquet(f"{outPath}/{lc}_rte/w_features-new.pq")
    features = features[
        (features.hdim_1 > n)
        & (features.hdim_2 > n)
        & (features.hdim_1 < (2230 - n))
        & (features.hdim_2 < (2150 - n))
    ]
    features = features[
        (features.time >= pd.to_datetime("2019-09-16-20"))
        & (features.time <= pd.to_datetime("2019-09-21-20"))
    ]

    features = features.drop_duplicates(["frame", "x", "y", "z"])

    tracks = tobac.linking_trackpy(
        features,
        None,
        dt=60 * 5,  # time in seconds separating each frame
        dxy=150,
        vertical_coord="ztn",
        **params,
    )
    tracks = tracks[tracks.cell != -1]
    tracks["lifetime"] = tracks.groupby("cell")["time_cell"].transform("max")

    tracks = tracks[tracks.lifetime >= dt.timedelta(minutes=15)]

    tracks.to_parquet(f"{outPath}/{lc}_rte/w_tracks-new.pq")
