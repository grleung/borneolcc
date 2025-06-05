import glob
import os 
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
from shared_plotting import *
import dask.distributed as dd

client = dd.Client('downdraft:8786')
client.upload_file("shared_model_params.py")

from shared_model_params import (
    get_rams_output,combine_tobac_list
)

def get_overlap_features(time, tracks_sub):

    cond_mask = xr.open_dataset(f'/squall/gleung/borneolcc-analysis/tobac/{run}_rte/cond_masks_column_anvil/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5',
                        chunks='auto',engine='h5netcdf')
    w_mask = xr.open_dataset(f'/squall/gleung/borneolcc-analysis/tobac/{run}_rte/w_masks/a-L-{time.strftime('%Y-%m-%d-%H%M%S')}.h5',
                            chunks='auto',engine='h5netcdf')

    # mask will have 1s where both updraft and condensate are present
    # This will allow us to make sure updraft and condensate regions are coincident/overlapping in space

    full_mask = (w_mask.segmentation_mask>0)*(cond_mask.segmentation_mask>0)
    
    fts = cond_mask.where(full_mask).segmentation_mask.compute()
    ftlist = np.unique(fts.values)
    ftlist = ftlist[~np.isnan(ftlist)]
    
    print(len(tracks_sub))
    
    tracks_sub = tracks_sub[tracks_sub.feature.isin(ftlist)]

    print(len(tracks_sub))

    return(tracks_sub)


run = 'lc2019'
tobacPath = f'/squall/gleung/borneolcc-analysis/tobac/{run}_rte/'


tracks = pd.read_parquet(f"{tobacPath}/w_tracks-new.pq")

cond = []
for p in sorted(glob.glob(f"{tobacPath}/cond_column_anvil_segmentation_*.pq")):
    df = pd.read_parquet(p)

    cond.append(df)

cond = combine_tobac_list(cond)

cond = cond.groupby(["time", "x", "z"]).last().reset_index()
cond["ncells_cond"] = cond["ncells"]

w = []
for p in sorted(glob.glob(f"{tobacPath}/w_segmentation_*.pq")):

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

times = sorted(tracks.time.unique())

print(len(times), len(times)//24)
for i, times_ in enumerate(np.array_split(times,len(times)//24)):

    if (i>=48) and (not os.path.exists(f'{tobacPath}/cloud_anvil_tracks_cleaned_wcond_{str(i).zfill(2)}.pq')):
        tracks_ = client.map(get_overlap_features, times_, [tracks[tracks.time==t] for t in times_])

        tracks_ = client.gather(tracks_)

        tracks_ = pd.concat(tracks_)

        tracks_.to_parquet(f'{tobacPath}/cloud_anvil_tracks_cleaned_wcond_{str(i).zfill(2)}.pq')

        print(f'{tobacPath}/cloud_anvil_tracks_cleaned_wcond_{str(i).zfill(2)}.pq')
