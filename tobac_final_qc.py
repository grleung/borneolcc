"""
Step 10 (last) of tobac processing: Final QC

Input: parquet file with merged features 'raining_cloudy_updraft_statistics.pq' + parquet file with merged features 'cloudy_updraft_statistics_full.pq'
Output: parquet file with features 'qc_final_cloudy_updrafts.pq'
"""

import os
import pandas as pd
import numpy as np
import xarray as xr

from shared_model_params import get_rams_landcover, nx, ny, bxy

modeldataPath = "/squall/gleung/borneolcc/"  # raw model output
tobacPath = "/squall/gleung/borneolcc-analysis/tobac/"

for run in ["lc1960", "lc2019"]:
    if (
        os.path.exists(f"{tobacPath}/{run}_rte/raining_cloudy_updraft_statistics.pq")
    ) and (os.path.exists(f"{tobacPath}/{run}_rte/cloudy_updraft_statistics_full.pq")):
        left = pd.read_parquet(
            f"{tobacPath}/{run}_rte/raining_cloudy_updraft_statistics.pq"
        )
        right = pd.read_parquet(
            f"{tobacPath}/{run}_rte/cloudy_updraft_statistics_full.pq"
        )

        # merge the two statistics files
        out = pd.merge(
            right,
            left[
                [
                    "feature",
                    "pcp_ncells",
                    "cellmax_pcpncells",
                    "pcp_area",
                    "pcp_mean",
                    "pcp_max",
                    "pcp_total",
                ]
            ].fillna(0),
            on="feature",
            suffixes=(False, False),
        )

        # assign max/min/initial CTH/CBH over cell lifetime
        out["cellmax_CTH"] = out.groupby("cell").CTH.transform("max")
        out["cellmin_CBH"] = out.groupby("cell").CBH.transform("min")
        out["cellinit_CBH"] = out.groupby("cell").CBH.transform("first")
        out["cellinit_ztn"] = out.groupby("cell").ztn.transform("first")
        out["cellmin_ztn"] = out.groupby("cell").ztn.transform("min")

        # removing all cells which initiated above the boundary layer (~>2.5km for conservative estimate)
        out = out[(out.cellmin_ztn <= 2500)]

        # remove any clouds which spend a significant fraction of lifetime as fog, where fog = cloud base touches the surface
        out["fog_flag"] = out.CBH < 0

        out = out[
            out.groupby("cell").fog_flag.transform("sum")
            < (0.25 * (out.groupby("cell").fog_flag.transform("count")))
        ]

        # read in land cover and assign each cell to land cover
        lc = get_rams_landcover(
            f"{modeldataPath}/{run}/rte/a-A-2019-09-16-140000-g1.h5"
        ).sel(x=slice(bxy, nx - bxy), y=slice(bxy, ny - bxy))
        lc = lc.assign(
            {"x": lc.x + 50, "y": lc.y + 50}
        )  # needs to be renumbered because of slicing to correspond with tobac parquet files
        lc_df = lc.lc.to_dataframe()

        out["lc"] = out.set_index(
            [np.floor(out.y).astype(int), np.floor(out.x).astype(int)]
        ).index.map(lc_df.lc)

        # assign land cover type initiated over
        out["cellinit_lc"] = out.groupby("cell").lc.transform("first")

        # remove all cells which initiated over water - not part of our sample
        out = out[out.cellinit_lc != 0]

        out["hour_day"] = out.local_time.dt.hour + out.local_time.dt.minute / 60

        print(run)
        # finally save file!
        out.to_parquet(f"{tobacPath}/{run}_rte/qc_final_cloudy_updrafts.pq")
