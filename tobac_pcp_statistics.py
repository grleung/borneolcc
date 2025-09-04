import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import (
    sum_labels,
    maximum,
    mean,
)
import dask.distributed as dd
import glob

client = dd.Client("solvarg:8786")
client.upload_file("shared_model_params.py")

from shared_model_params import get_rams_output

client.upload_file("shared_processing.py")
from shared_processing import (
    compute_pcp,
)


dxy = 150


def get_masked_statistics(sub, dataPath, tobacPath):
    if len(sub) > 0:
        fts = sub.feature.unique()
        time = pd.to_datetime(sub.timestr.iloc[0])

        ds = get_rams_output(
            f"{dataPath}/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}-g1.h5",
            variables=[
                "PCPRR",
                "PCPRP",
                "PCPRS",
                "PCPRA",
                "PCPRG",
                "PCPRH",
                "PCPRD",
            ],
        )

        pcp = compute_pcp(ds)

        pcp_mask = xr.open_dataset(
            f"{tobacPath}/pcp_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5",
            engine="h5netcdf",
            chunks="auto",
        )

        sub["pcp_area"] = (
            sum_labels(
                (pcp_mask.segmentation_mask / pcp_mask.segmentation_mask),
                pcp_mask.segmentation_mask,
                fts,
            )
            * dxy
            * dxy
            / (1000 * 1000)
        )

        print("pcp stats")

        sub["pcp_mean"] = mean(
            pcp,
            pcp_mask.segmentation_mask,
            fts,
        )

        sub["pcp_max"] = maximum(
            pcp,
            pcp_mask.segmentation_mask,
            fts,
        )

        sub["pcp_total"] = sub.pcp_mean * sub.pcp_area

        print("pcp2 stats")

        return sub
    else:
        print(sub)


n = 24

for lc in ["lc1960", "lc2019"]:
    dataPath = f"/squall/gleung/borneolcc/{lc}/rte/"
    tobacPath = f"/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/"

    tracks = pd.read_parquet(f"{tobacPath}/cloudy_updrafts_raining.pq")
    frames = tracks.frame.unique()
    times = tracks.time.unique()

    for i, frames_ in enumerate(
        np.array_split(sorted(frames), len(frames) // n)
    ):

        if not os.path.exists(
            f"{tobacPath}/raining_cloudy_updraft_statistics_{str(i).zfill(2)}.pq"
        ):
            x = client.map(
                get_masked_statistics,
                [tracks[tracks.frame == frame] for frame in frames_],
                dataPath=dataPath,
                tobacPath=tobacPath,
            )
            x = client.gather(x)

            x = pd.concat(x)

            x.to_parquet(
                f"{tobacPath}/raining_cloudy_updraft_statistics_{str(i).zfill(2)}.pq"
            )

    savePaths = sorted(
        glob.glob(f"{tobacPath}/raining_cloudy_updraft_statistics_*.pq")
    )
    print(len(savePaths))
    print(len(frames) // 24)

    # make sure all the saved files are present
    if len(savePaths) == (len(frames) // n):
        # read in all the files, combine, and save

        all_df = []
        for p in savePaths:
            all_df.append(pd.read_parquet(p, engine="pyarrow"))
        all_df = pd.concat(all_df)

        all_df.to_parquet(f"{tobacPath}/raining_cloudy_updraft_statistics.pq")

        print(len(all_df.time.unique()))