from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from palettable.cmocean.sequential import Ice_20
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy
import cartopy.crs as ccrs

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

#style = "/home/gleung/scripts/styles/bee-presentationtransparent.mplstyle"
#bg = "#2E3745"
#lcol = "white"
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


def add_latlon(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

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