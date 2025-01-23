import pandas as pd
import os
import glob
import numpy as np
import dask.distributed as dd
import datetime as dt


client = dd.Client("snowfall2:8786")
client.upload_file("shared_model_params.py")

from shared_model_params import (
    get_rams_output,
    combine_tobac_list,
    save_files,
    compute_cond,
)


modelPath = "/squall/gleung/borneolcc/"
outPath = f"/squall/gleung/borneolcc-analysis/tobac/"

for lc in ["lc1960"]:
    tracks = pd.read_parquet(f"{outPath}/{lc}_rte/w_tracks-new.pq")

    cond = []
    for p in sorted(glob.glob(f"{outPath}/{lc}_rte/cond_segmentation_*.pq")):

        df = pd.read_parquet(p)

        cond.append(df)

    cond = combine_tobac_list(cond)

    cond = cond.groupby(["time", "x", "z"]).last().reset_index()
    cond["ncells_cond"] = cond["ncells"]

    w = []
    for p in sorted(glob.glob(f"{outPath}/{lc}_rte/w_segmentation_*.pq")):

        df = pd.read_parquet(p)

        w.append(df)

    w = combine_tobac_list(w)
    w = w.groupby(["time", "x", "z"]).last().reset_index()
    w["ncells_w"] = w["ncells"]

    tracks = tracks.set_index(["time", "cell"])

    tracks["w_ncells"] = tracks.index.map(
        w.groupby(["time", "cell"]).ncells_w.first()
    )
    tracks["cond_ncells"] = tracks.index.map(
        cond.groupby(["time", "cell"]).ncells_cond.first()
    )

    tracks["cellmax_wncells"] = tracks.groupby("cell").w_ncells.transform("max")
    tracks["cellmax_condncells"] = tracks.groupby("cell").cond_ncells.transform(
        "max"
    )

    print("Started with", len(tracks))

    tracks = tracks[
        (tracks.cellmax_wncells > 0) & (tracks.cellmax_condncells > 0)
    ]

    print("Ended with", len(tracks))

    tracks = tracks.reset_index()

    tracks["local_time"] = tracks.time + dt.timedelta(hours=8)

    tracks.to_parquet(f"{outPath}/{lc}_rte/combined_cond-w_segmented_tracks.pq")
