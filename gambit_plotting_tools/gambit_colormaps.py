"""
Color maps
==========

Dark color map from pippi-generated ctioga2 script: #000--#000(0.04599999999999993)--#33f(0.31700000000000006)--#0ff(0.5)--#ff0

Light color map wasn't defined in pippi/ctioga.
"""

import matplotlib


hex_colors = ["#000", "#33f", "#0ff", "#ff0"]
rgb_colors = [matplotlib.colors.hex2color(
    hex_color) for hex_color in hex_colors]

gambit_dark_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gambit-dark", rgb_colors)


gambit_light_cmap = matplotlib.pyplot.get_cmap("magma_r")
gambit_light_cmap.name = "gambit-light"


def register_cmaps():
    """
    Register gambit colormaps by name, so that they are accessible by e.g. cmap="gambit-light"
    """
    matplotlib.colormaps.register(gambit_dark_cmap)
    matplotlib.colormaps.register(gambit_light_cmap)