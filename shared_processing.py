import pandas as pd
import numpy as np
import xarray as xr
from shared_model_params import assign_dz, lv, cp, p00, rd


def compute_pcp(ds):
    ds = ds.assign(
        PCPTOT=3600
        * (
            ds.PCPRR
            + ds.PCPRP
            + ds.PCPRS
            + ds.PCPRA
            + ds.PCPRG
            + ds.PCPRH
            + ds.PCPRD
        )
    )
    ds = ds["PCPTOT"]
    return ds


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


def get_land_mean(ds: xr.Dataset, landmask: xr.Dataset, profile=False) -> xr.Dataset:
    """
    Given a RAMS array, takes the mean value among land points only

    Arguments:
        ds -- xarray dataset of variables from RAMS
        landmask -- landmask of same size as ds

    Returns:
        Dataset of mean values among land grid points only
    """
    ds = ds * landmask
    if profile:
        return ds.sum(dim=('x','y')) / landmask.sum()
    else:
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
    ds = ds.assign(
        {
            "lhf": -ds.SFLUX_R * lv,  # latent heat flux
            "shf": -ds.SFLUX_T * cp,  # sensible heat flux
        }
    )

    # take first real model level above surface
    ds = ds.assign(
        {
            "lwdn": ds.LWDN.sel(z=1),  # downwelling longwave
            "lwup": ds.LWUP.sel(z=1),  # upwelling longwave
            "swdn": ds.SWDN.sel(z=1),  # downwelling shortwave
            "swup": ds.SWUP.sel(z=1),  # upwelling shortwave
        }
    )

    ds = ds.assign(
        {
            "swnet": ds.swdn - ds.swup,  # net shortwave
            "lwnet": ds.lwdn - ds.lwup,  # net longwave
        }
    )

    ds = ds.assign(g=-(ds.swnet + ds.lwnet + ds.shf + ds.lhf))  # heat storage

    return ds[["lhf", "shf", "lwnet", "swnet", "g"]]


def compute_canopy_nearsurf(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes temperature and dewpoints at nearest model level to the surface
    and at the canopy height for given RAMS file

    Arguments:
        ds -- RAMS output with variables [THETA,PI,RV, CAN_TEMP,CAN_RVAP]

    Returns:
        xarray of temp and dewpoint at near-surface atmosphere and canopy
    """
    from shared_model_params import cp, rd, p00
    import metpy.calc as mpcalc
    import metpy.units as units

    ds = ds.sel(z=1, p=1)

    ds = ds.assign(
        {
            "AIR_T": (ds.THETA * (ds.PI / cp)) - 273.15,  # temperature in degC
            "PRES": (p00 * (ds.PI / cp) ** (cp / rd)) / 100,  # pressure in hPa
        }
    )

    ds = ds.assign(
        AIR_Td=(
            ("y", "x"),
            mpcalc.dewpoint_from_specific_humidity(
                ds.PRES.values * units.units("hPa"),
                ds.AIR_T.values * units.units("degC"),
                ds.RV.values * units.units("kg/kg"),
            ).magnitude,
        )
    )  # dewpoint in degC

    ds = ds.assign(AIR_RV=ds.RV)

    ds = ds.assign(CAN_T=ds.CAN_TEMP - 273.15)  # canopy temp in degC
    ds = ds.assign(
        CAN_Td=(
            ("y", "x"),
            mpcalc.dewpoint_from_specific_humidity(
                ds.PRES.values * units.units("hPa"),
                ds.CAN_T.values * units.units("degC"),
                ds.CAN_RVAP.values * units.units("kg/kg"),
            ).magnitude,
        )
    )  # canopy dewpoint in degC

    ds = ds.assign(CAN_RV=ds.CAN_RVAP)

    return ds[["AIR_T", "AIR_Td", "AIR_RV", "CAN_T", "CAN_Td", "CAN_RV"]]


def compute_surf_pert(
    ds: xr.Dataset, landmask: xr.Dataset, topo: xr.Dataset
) -> xr.Dataset:
    """
    Computes perturbations from mean over land for given RAMS file. Currently
    calculates heat flux (LHF + SHF) and near surface thetav (as a measure of buoyancy)

    Arguments:
        ds -- RAMS output with variables [SFLUX_R,SFLUX_T,THETA,RV]
        landmask -- xarray with landmask
        topo -- xarray with topography height

    Returns:
        xarray(y,x) of heat fluxes (hf) and thetav + perturbation from mean values
    """
    from shared_model_params import lv, cp

    ds = ds.sel(z=1)

    # only need land points where altitude < 500m ASL
    ds = ds.where(landmask).where(topo <= 500)

    # compute virtual potential temperature
    ds = ds.assign(thetav=ds.THETA * (1 + (0.61 * ds.RV)))

    # convert temperature and moisture flux into W/m2 (magnitude only)
    ds = ds.assign(
        {
            "lhf": ds.SFLUX_R * lv,  # latent heat flux
            "shf": ds.SFLUX_T * cp,  # sensible heat flux
        }
    )
    ds = ds.assign(hf=ds.lhf + ds.shf)

    # get perturbation from mean value at a given time
    ds = ds.assign(hf_pert=ds.hf - (ds.hf.mean(dim=("x", "y"))))
    ds = ds.assign(thetav_pert=ds.thetav - (ds.thetav.mean(dim=("x", "y"))))

    return ds[["hf", "hf_pert", "thetav", "thetav_pert"]]


from shared_model_params import get_rams_landcover, remove_boundaries, bxy

past_lc = get_rams_landcover(
    f"/squall/gleung/borneolcc/lc1960/rte/a-A-2019-09-16-140000-g1.h5"
)
past_lc = remove_boundaries(past_lc, bxy=bxy)
pres_lc = get_rams_landcover(
    f"/squall/gleung/borneolcc/lc2019/rte/a-A-2019-09-16-140000-g1.h5"
)
pres_lc = remove_boundaries(pres_lc, bxy=bxy)


forest_loss = ((pres_lc.lc == 7) / (pres_lc.lc != 0)) - (
    (past_lc.lc == 7) / (past_lc.lc != 0)
)


def smooth_data_plotting(
    data, coarseres, rollres, coarseagg="mean", min_periods=1
):
    data = (
        data.coarsen(x=coarseres, y=coarseres, boundary="pad")
        .reduce(coarseagg)
        .rolling(x=rollres, y=rollres, min_periods=min_periods, center=True)
        .mean()
    )

    return data

def rolling_cycle(d, window=3):
    return (
        pd.Series(
            np.concatenate([d[-int(window / 2) :], d, d[: int(window / 2)]])
        )
        .rolling(window, center=True)
        .mean()
        .dropna()
        .values
    )

def compute_thermo_prof(ds, zslice=slice(1, 35)):
    """
    Computes temperature, dewpoint, and pressure for nearsurface profile
    for given RAMS file

    Arguments:
        ds -- RAMS output with variables [THETA,PI,RV]

    Returns:
        xarray of temp, dewpoint, pressure at given profile levels
    """
    from shared_model_params import cp, rd, p00
    import metpy.calc as mpcalc
    import metpy.units as units

    # select only the z levels specified
    ds = ds.sel(z=zslice)

    # calculations of pressure (hPa), temp (degC)
    ds = ds.assign(
        {
            "PRES": (p00 * (ds.PI / cp) ** (cp / rd)) / 100,
            "TEMP": ds.THETA * (ds.PI / cp) - 273.15,
        }
    )

    ds = ds.assign(
        Td=(
            ("z","y", "x"),
            mpcalc.dewpoint_from_specific_humidity(
                ds.PRES.values * units.units("hPa"),
                ds.TEMP.values * units.units("degC"),
                ds.RV.values * units.units("kg/kg"),
            ).magnitude,
        )
    )  # dewpoint in degC


    return(ds[['TEMP','PRES','Td']])






    
