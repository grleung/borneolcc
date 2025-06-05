import pandas as pd
import numpy as np
import xarray as xr
from shared_model_params import assign_dz, lv, cp


def compute_cond(
    ds: xr.Dataset, return_dens: bool = False, cloud: bool = True
) -> xr.Dataset:
    """
    Computes condensate mixing ratio (and density) from RAMS output

    Arguments:
        ds (xr.Dataset) -- RAMS output from one timestep, should have PI, THETA, RV + either (RCP, RSP, RPP) or (RTP)

    Keyword Arguments:
        return_dens (bool) -- Flag to return density (default: {False})
        cloud (bool) -- Should condensate be calculated as (cloud + snow + pristine ice) or as (total water - vapor)?
                        The latter includes all hydrometeors including precipitation. (default: {True})

    Returns:
        RAMS xarray dataset with condensate mixing ratio (and density)
    """
    ds = ds.assign(PRES=p00 * (ds.PI / cp) ** (cp / rd))
    ds = ds.assign(TEMP=ds.THETA * (ds.PI / cp))
    ds = ds.assign(DENS=ds.PRES / (rd * ds.TEMP * (1 + (0.61 * ds.RV))))

    if cloud:
        ds = ds.assign(COND=(ds.RCP + ds.RSP + ds.RPP) * ds.DENS)
    else:
        ds = ds.assign(COND=(ds.RTP - ds.RV) * ds.DENS)

    if return_dens:
        ds = ds[["COND", "DENS"]]
    else:
        ds = ds["COND"]
    return ds


def compute_intcond(ds: xr.Dataset, use_dens=True) -> xr.Dataset:
    """
    Vertically integrate condensate mixing ratio to get integrated condensate (mm)

    Arguments:
        ds (xr.Dataset) -- RAMS xarray dataset with COND (and possibly DENS)

    Keyword Arguments:
        use_dens (bool) -- flag for incorporating density in calculation (more accurate, technically) (default: {True})

    Returns:
        RAMS xarray dataset with integrated condensate (mm)
    """

    ds = assign_dz(ds)

    if use_dens:
        ds = ds.assign(intCON=((ds.DENS * ds.COND * ds.dz).sum(dim="z")) + 1e-9)
    else:
        ds = ds.assign(intCON=((ds.COND * ds.dz).sum(dim="z")) + 1e-9)

    return ds["intCON"]


def find_paths_in_time_range(
    run: str,
    center_time: float,
    duration: float,
    modeldataPath="/squall/gleung/borneolcc/",
    localtime=True,
) -> list:
    """
    Returns a list of all RAMS output files in a directory that are within [duration] of [center time],
    regardless of the date (e.g., within 0.5hrs of 4:00).

    Arguments:
        run (str) -- name of the run, which is the subfolder within modeldataPath
        center_time (float) -- central time, in hours
        duration (float) -- duration to search in either direction, in hours

    Keyword Arguments:
        modeldataPath (str) -- _description_ (default: {"/squall/gleung/borneolcc/"})
        localtime (bool) -- flag for finding center_time in local time (default: True)

    Returns:
        List of paths within given timerange.
    """

    import glob
    import datetime as dt

    paths = sorted(glob.glob(f"{modeldataPath}{run}/rte/a-L-*-g1.h5"))[
        6 * 12 : ((6 * 12) + (3 * 24 * 12)) + 1
    ]  # first 6 hours are spinup; analysis period is first 3 days after spinup period

    times_utc = [pd.to_datetime(p.split("/")[-1][4:-6]) for p in paths]
    times_local = [t + dt.timedelta(hours=8) for t in times_utc]

    paths = pd.DataFrame(
        [paths, times_utc, times_local],
        index=["path", "time_utc", "time_local"],
    ).T
    paths["time_utc"] = pd.to_datetime(paths.time_utc)
    paths["time_local"] = pd.to_datetime(paths.time_local)

    paths["hour_day"] = (
        paths.time_local.dt.hour + paths.time_local.dt.minute / 60
    )

    if center_time == 0:
        paths = paths[
            (np.abs(paths.hour_day - center_time) < duration)
            | (np.abs(paths.hour_day - 24) < duration)
        ]
    else:
        paths = paths[np.abs(paths.hour_day - center_time) < duration]

    return paths.path.values


def get_land_mean(ds: xr.Dataset, landmask: xr.Dataset) -> xr.Dataset:
    """
    Given a RAMS array, takes the mean value among land points only

    Arguments:
        ds -- xarray dataset of variables from RAMS
        landmask -- landmask of same size as ds

    Returns:
        Dataset of mean values among land grid points only
    """
    ds = ds * landmask
    return ds.sum() / landmask.sum()


def compute_seb(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes surface energy budget terms for given RAMS file
    Sign convection is positive for terms with energy into the surface
    negative for terms with energy out of the surface.

    Arguments:
        ds -- RAMS output with variables [SFLUX_R,SFLUX_T,LWUP,LWDN,SWUP,SWDN]

    Returns:
        xarray of surface energy budget terms
    """
    from shared_model_params import lv, cp

    # convert temperature and moisture flux into W/m2
    ds = ds.assign(lhf=-ds.SFLUX_R * lv)  # latent heat flux
    ds = ds.assign(shf=-ds.SFLUX_T * cp)  # sensible heat flux

    # take first real model level above surface
    ds = ds.assign(lwdn=ds.LWDN.sel(z=1))  # downwelling longwave
    ds = ds.assign(lwup=ds.LWUP.sel(z=1))  # upwelling longwave
    ds = ds.assign(swdn=ds.SWDN.sel(z=1))  # downwelling shortwave
    ds = ds.assign(swup=ds.SWUP.sel(z=1))  # upwelling shortwave

    ds = ds.assign(swnet=ds.swdn - ds.swup)  # net shortwave
    ds = ds.assign(lwnet=ds.lwdn - ds.lwup)  # net longwave

    ds = ds.assign(g=-(ds.swnet + ds.lwnet + ds.shf + ds.lhf))  # heat storage

    return ds[["lhf", "shf", "lwnet", "swnet", "g"]]
