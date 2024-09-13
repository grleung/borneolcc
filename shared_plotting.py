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

'''style = "/home/gleung/scripts/styles/bee-presentationtransparent.mplstyle"
bg = "#2E3745"
lcol = "white"'''
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
