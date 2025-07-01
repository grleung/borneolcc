# Import some shared libraries
import dask.distributed as dd
import pandas as pd
import tobac
import datetime as dt
from shared_model_params import dx, nx, ny, bxy

outPath = "/squall/gleung/borneolcc-analysis/tobac/"
lcs = ["lc1960", "lc2019"]


# tobac tracking parameters
# see tobac documentation for more detailed description of each parameter
params = {}
params["extrapolate"] = 0
params["order"] = 1
params["memory"] = 0
params["time_cell_min"] = (
    15 * 60
)  # in seconds (minimum lifetime of 15 minutes = 4 points in time)
params["method_linking"] = "predict"
params["adaptive_step"] = 0.75
params["adaptive_stop"] = 1.0
params["d_max"] = (
    150 * 10
)  # this is the distance (in m) of the search radius around trackpy predictive track

for lc in lcs:
    print(lc)
    features = pd.read_parquet(f"{outPath}/{lc}_rte/w_features.pq")

    # exclude any features which are too close to the boundary of the domain
    features = features[
        (features.hdim_1 > bxy)
        & (features.hdim_2 > bxy)
        & (features.hdim_1 < (ny - bxy))
        & (features.hdim_2 < (nx - bxy))
    ]

    tracks = tobac.linking_trackpy(
        features,
        None,
        dt=60 * 5,  # time in seconds separating each frame
        dxy=dx,
        vertical_coord="ztn",
        **params,
    )

    # remove untracked features
    tracks = tracks[tracks.cell != -1]

    # new column for cell lifetime in minutes
    tracks["lifetime"] = tracks.groupby("cell")["time_cell"].transform(
        "max"
    ) / dt.timedelta(minutes=1)

    tracks.to_parquet(f"{outPath}/{lc}_rte/w_tracks.pq")
