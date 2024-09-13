import pandas as pd
import numpy as np
import h5py
import xarray as xr

# physical constants
g = 9.8065
eps = 0.622
cp = 1004
rd = 287
p00 = 100000
rgas = 287
lv = 2.5e6


nx = 2750  # number x pts
dx = 150  # grid spacing
ny = 2802  # number y pts
dy = 150  # grid spacing

b = 100
x = nx * dx / 1000  # total x-length
y = ny * dy / 1000  # total y-length
nz = 106  # number z pts

rams_dims_lite = {
    "phony_dim_0": "p",
    "phony_dim_3": "z",
    "phony_dim_1": "y",
    "phony_dim_2": "x",
}

rams_dims_anal = {
    "phony_dim_2": "z",
    "phony_dim_0": "y",
    "phony_dim_1": "x",
    "phony_dim_3": "p",  # patch
    "phony_dim_4": "s",  # surf water
    "phony_dim_5": "g",  # soil levels
}


def get_rams_output(
    path,
    variables,
    dims=rams_dims_lite,
):

    drop_var = [v for v in all_var if v not in variables]

    ds = xr.open_dataset(
        path,
        phony_dims="access",
        engine="h5netcdf",
        chunks="auto",
        drop_variables=drop_var,
    )

    ds = rename_dims(ds, dims)

    if len(variables) == 1:
        ds = ds[variables[0]]

    return ds


def rename_dims(ds, dims=rams_dims_lite):
    return ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))


def read_rams_file(p, in_vars, dims):
    ds = xr.open_dataset(
        p, phony_dims="access", engine="h5netcdf", chunks="auto"
    )[in_vars]

    ds = ds.rename_dims(dict([(d, dims.get(d)) for d in ds.dims]))

    return ds


def read_header(dataPath, p, nz, var="__ztn01", varname="z"):
    # fxn to read thermodynamic z from header file
    header_file_name = (
        f"{dataPath}/{p.split('/')[-1].split('.')[0][:-2]}head.txt"
    )
    with open(header_file_name) as f:
        mylist = f.read().splitlines()
    ix = mylist.index(var)
    numlines = int(mylist[ix + 1])
    coord = mylist[ix + 2 : ix + 2 + numlines]
    coord = np.array([float(x) for x in coord])
    return coord


zg = read_header(
    "/squall/gleung/borneolcc/lc1960/rte/",
    f"a-A-2019-09-16-140000-g1.h5",
    11,
    var="__slz",
    varname="zg",
)

zm = read_header(
    "/squall/gleung/borneolcc/lc1960/rte/",
    f"a-A-2019-09-16-140000-g1.h5",
    nz,
    var="__zmn01",
    varname="zm",
)
zt = read_header(
    "/squall/gleung/borneolcc/lc1960/rte/",
    f"a-A-2019-09-16-140000-g1.h5",
    nz,
)
alt = zt
dz = 1 / read_header(
    "/squall/gleung/borneolcc/lc1960/rte/",
    f"a-A-2019-09-16-140000-g1.h5",
    nz,
    var="__dztn01",
    varname="dz",
)


def assign_topt(ds):
    # topography height 2d
    # reads topography height
    file = h5py.File(
        f"/squall/gleung/borneolcc/lc1960/rte/a-A-2019-09-16-140000-g1.h5",
        "r",
    )
    topt = file["TOPT"][:, :]
    file.close()

    ds = ds.assign(topt=(("y", "x"), topt))
    return ds


def assign_alt(ds):
    # altitudes of sigma-z levels in 3d
    ds = ds.assign(alt=(("z", alt)))
    return ds


def assign_dz(ds):
    # altitudes of sigma-z levels in 3d
    ds = ds.assign(dz=(("z", dz)))
    return ds


lite_var = [
    "CAN_RVAP",
    "CAN_TEMP",
    "CAP",
    "CCP",
    "CDP",
    "CGP",
    "CHP",
    "CPP",
    "CRP",
    "CSP",
    "FTHRD",
    "FTHRDLW",
    "FTHRDSW",
    "GLAT",
    "GLON",
    "GROUND_RSAT",
    "GROUND_RVAP",
    "LWDN",
    "LWUP",
    "PCPRA",
    "PCPRD",
    "PCPRG",
    "PCPRH",
    "PCPRP",
    "PCPRR",
    "PCPRS",
    "PI",
    "RAP",
    "RCP",
    "RDP",
    "RGP",
    "RHP",
    "RPP",
    "RRP",
    "RSP",
    "RSTAR",
    "RTP",
    "RV",
    "SFLUX_R",
    "SFLUX_T",
    "SOIL_ENERGY",
    "SOIL_WATER",
    "SWDN",
    "SWUP",
    "THETA",
    "TSTAR",
    "UP",
    "USTAR",
    "VEG_TEMP",
    "VEG_WATER",
    "VP",
    "WP",
]

ana_var = [
    "ACCPA",
    "ACCPD",
    "ACCPG",
    "ACCPH",
    "ACCPP",
    "ACCPR",
    "ACCPS",
    "ALBEDT",
    "AODT",
    "BEXT",
    "CAN_RVAP",
    "CAN_TEMP",
    "CAP",
    "CCP",
    "CDP",
    "CGP",
    "CHP",
    "CIFNP",
    "CN1MP",
    "CN1NP",
    "CN2MP",
    "CN2NP",
    "COSZ",
    "CPP",
    "CRP",
    "CSP",
    "DN0",
    "DPCPG",
    "FTHRD",
    "FTHRDLW",
    "FTHRDSW",
    "GLAT",
    "GLON",
    "GROUND_RSAT",
    "GROUND_RVAP",
    "HKH",
    "LEAF_CLASS",
    "LWDN",
    "LWUP",
    "PATCH_AREA",
    "PATCH_ROUGHM",
    "PATCH_ROUGHT",
    "PC",
    "PCPG",
    "PCPRA",
    "PCPRD",
    "PCPRG",
    "PCPRH",
    "PCPRP",
    "PCPRR",
    "PCPRS",
    "PCPVA",
    "PCPVD",
    "PCPVG",
    "PCPVH",
    "PCPVP",
    "PCPVR",
    "PCPVS",
    "PI",
    "PP",
    "Q2",
    "Q6",
    "Q7",
    "QPCPG",
    "RAP",
    "RCP",
    "RDP",
    "RGP",
    "RHKM",
    "RHP",
    "RLONG",
    "RLONGUP",
    "RLONTOP",
    "RPP",
    "RRP",
    "RSHORT",
    "RSP",
    "RSTAR",
    "RTP",
    "RV",
    "RVKH",
    "RVKM",
    "SFCWATER_DEPTH",
    "SFCWATER_ENERGY",
    "SFCWATER_MASS",
    "SFCWATER_NLEV",
    "SFLUX_R",
    "SFLUX_T",
    "SFLUX_U",
    "SFLUX_V",
    "SFLUX_W",
    "SOIL_ENERGY",
    "SOIL_ROUGH",
    "SOIL_TEXT",
    "SOIL_WATER",
    "STOM_RESIST",
    "SWDN",
    "SWUP",
    "THETA",
    "THP",
    "TOPT",
    "TOPZO",
    "TSTAR",
    "UC",
    "UP",
    "USTAR",
    "VC",
    "VEG_ALBEDO",
    "VEG_FRACAREA",
    "VEG_HEIGHT",
    "VEG_LAI",
    "VEG_NDVIC",
    "VEG_NDVIF",
    "VEG_NDVIP",
    "VEG_ROUGH",
    "VEG_TAI",
    "VEG_TEMP",
    "VEG_WATER",
    "VKH",
    "VP",
    "WC",
    "WP",
]
