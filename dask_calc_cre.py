import pandas as pd
import xarray as xr
import numpy as np
import os 
import glob
import dask.distributed as dd
import datetime as dt
import h5netcdf

client = dd.Client('snowfall2:8786')

client.upload_file("shared_model_params.py")
from shared_model_params import (
    get_rams_output,
    rams_dims_anal,
)

runs = ['lc1960','lc2019']
# copy Past Land Cover from RAMS into past_lc dataframe (this is constant over all times so just need one)
run = 'lc1960'
# these 2d things are fixed in time
coord = get_rams_output(f"/squall/gleung/borneolcc/{run}/rte/a-A-2019-09-16-140000-g1.h5",
    variables=[
        "PATCH_AREA",
        "LEAF_CLASS",
    ],
    dims=rams_dims_anal)[[
        "PATCH_AREA",
        "LEAF_CLASS",
    ]]

coord = coord.assign(LCTYPE=(coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim="p"))
coord["lc"] = (coord.LEAF_CLASS * coord.PATCH_AREA).sum(dim=("p"))

landmask = (coord.lc!=0).compute()


def calc_cres(p,fts,lc='lc1960'):
    ds = get_rams_output(p,variables=['SWDN','SWUP','LWUP']).sel(z=105)

    cmask = xr.open_dataset(f'/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/cond_masks_column_anvil/{p.split('/')[-1][:-6]}.h5')
    clr_mask = (cmask.segmentation_mask!=0).max(dim='z')
    cld_mask = (cmask.segmentation_mask.isin(fts)).max(dim='z')

    ds = ds.assign(clr_mask = clr_mask)
    ds = ds.assign(cld_mask = cld_mask)
    ds = ds.where(landmask)

    ds = ds.assign(lwnet = -ds.LWUP)
    ds = ds.assign(swnet = ds.SWDN-ds.SWUP)
    ds = ds.assign(rsnet = ds.swnet+ds.lwnet)

    ds = ds[['lwnet','swnet','rsnet','cld_mask','clr_mask']].sel(x=slice(50,2100),y=slice(50,2180))

    cld = (ds.where(ds.cld_mask!=0))[['lwnet','swnet','rsnet']].mean()
    clr = (ds.where(ds.clr_mask==0))[['lwnet','swnet','rsnet']].mean()

    cre = (cld - clr)
    out = xr.concat([clr,cld,cre],dim=pd.Index(['clr','cld','cre'],name='type')).compute()
    return(out)

hourss = np.arange(0,(24*3)-8,5/60)

for lc in ['lc1960']:
    feats = pd.read_parquet(f'/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/final_cells_land_nofog.pq')

    for i,hours in enumerate(np.array_split(hourss,36)):
        paths = [f"/squall/gleung/borneolcc/{lc}/rte/a-L-{(pd.to_datetime(f'2019-09-17-040000')+
                                                        dt.timedelta(hours=h.astype('float'))).strftime('%Y-%m-%d-%H%M%S')}-g1.h5"
                                                            for h in hours]
        paths = [p for p in paths if os.path.exists(f'/squall/gleung/borneolcc-analysis/tobac/{lc}_rte/cond_masks_column_anvil/{p.split('/')[-1][:-6]}.h5')]

        times =  [pd.to_datetime(p.split('/')[-1][4:-6]) for p in paths]
        fts = [feats[feats.time==t].feature.unique() for t in times]

        paths, fts,times =zip(*[(p,f,t) for p,f,t in zip(paths,fts,times) if len(f)>0])

        print(len(paths))

        ds = client.map(calc_cres,
                        paths,
                        fts,
                        lc=lc)
        ds = client.gather(ds)

        ds = xr.concat(ds,dim='time')

        ds.to_netcdf(f'/squall/gleung/borneolcc-analysis/toa-fluxes/{lc}_full_{str(i).zfill(2)}.h5',engine='h5netcdf',)