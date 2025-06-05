from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from palettable.cmocean.sequential import Ice_20
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import xarray as xr

from palettable.cartocolors.qualitative import Prism_8

prism = Prism_8.mpl_colors
blue = prism[1]
green = prism[3]
purple = prism[0]
red = prism[7]
orange = prism[6]
yellow = prism[5]
gray = "#303039"

# importing plotting parameters
style = "/home/gleung/scripts/styles/bee-paperlight.mplstyle"
bg = "white"
lcol = "black"

# style = "/home/gleung/scripts/styles/bee-presentationtransparentdark.mplstyle"
# bg = "#2E3745"
# lcol = "white"
plt.style.use(style)

cloud = mcolors.ListedColormap(Ice_20.mpl_colors[4:])
cloud.set_bad(Ice_20.mpl_colors[4])
cloud.set_under(Ice_20.mpl_colors[4])


# Add every font at the specified location
font_dir = ["/home/gleung/scripts/futura"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set font family globally
rcParams["font.family"] = "Futura"


def add_latlon(ax: plt.Axes) -> None:
    """
    Formatter for pretty latitude/longitude labels

    Arguments:
        ax (plt.Axes) -- axis to label

    Returns:
        None
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return None


def add_legend(
    ax,
    handles="None",
    labels="None",
    ncols=None,
    loc=None,
    bbox_to_anchor=None,
    title=None,
    handlelength=0,
):

    if handles != "None":
        leg = ax.legend(
            handles=handles,
            labels=labels,
            ncols=ncols,
            loc=loc,
            bbox_to_anchor=bbox_to_anchor,
            title=title,
            handlelength=handlelength,
        )
    else:
        leg = ax.legend(
            bbox_to_anchor=bbox_to_anchor,
        )

    handles, labels = leg.axes.get_legend_handles_labels()
    texts = leg.get_texts()

    for (
        h,
        text,
    ) in zip(handles, texts):
        try:
            text.set_color(h.get_facecolor()[0])
        except:
            text.set_color(h.get_color())


def add_diurnal_annotation(ax: plt.Axes, zeroline: bool = True) -> None:
    """
    Plotting decorators for standard diurnal plots: shades nighttime area, adds axis labels

    Arguments:
        ax (plt.Axes) -- axis to plot on

    Keyword Arguments:
        zeroline (bool) -- add horizontal line at zero or not (default: {True})

    Returns:
        None
    """

    ax.set_xlabel("Hour of Day (Local Time)")
    ax.tick_params(axis="x", labelrotation=45)

    ax.axvspan(
        0,
        6.25,
        zorder=0,
        color="gray",
        alpha=0.2,
    )
    ax.axvspan(
        18.5,
        24,
        zorder=0,
        color="gray",
        alpha=0.2,
    )

    ax.set_xlim(0, 23.5)

    if zeroline:
        ax.axhline(0, zorder=0, lw=1, ls=":")

    return None


def add_topography(
    ax: plt.Axes, topo: xr.Dataset, alts: list[float] = [500], lcol: str = lcol
) -> None:
    """
    Plots topography from RAMS output on the given axis

    Arguments:
        ax (plt.Axes) -- axis to plot on
        topo (xr.Dataset) -- topography data from RAMS

    Keyword Arguments:
        alts (list[float])-- topopgraphy altitudes to plot (m ASL) (default: {[500]})
        lcol (str) -- line color (default: {lcol})

    Returns:
        None
    """

    ax.contour(
        topo.GLON,
        topo.GLAT,
        topo.TOPT,
        transform=ccrs.PlateCarree(),
        levels=alts,
        colors=lcol,
        linewidths=1,
    )

    return None
