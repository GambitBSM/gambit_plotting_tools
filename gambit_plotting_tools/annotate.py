"""
Add annotations to axis
=======================
"""

import matplotlib.pyplot as plt
import PIL
import numpy as np
import importlib.resources

from gambit_plotting_tools.gambit_plot_settings import plot_settings


def add_header(header_text, ax=None, fontsize=plot_settings["header_fontsize"]):
    if ax is None:
        ax = plt.gca()
    ax.set_title(
        header_text, 
        loc="right", 
        pad=plot_settings["header_pad"], 
        fontsize=fontsize)


def add_gambit_header(ax=None, version=None, fontsize=plot_settings["header_fontsize"]):
    if version is not None:
        if plt.rcParams.get("text.usetex"):        
            header_text = f"\\textsf{{GAMBIT {version}}}"
        else:
            header_text = f"GAMBIT {version}"
    else:
        if plt.rcParams.get("text.usetex"):        
            header_text = "\\textsf{GAMBIT}"
        else:
            header_text = "GAMBIT"
    add_header(header_text, fontsize=fontsize)


def load_small_gambit_logo():
    im = PIL.Image.open(importlib.resources.path('gambit_plotting_tools', 'gambit_logo_small.png'))
    im = np.array(im)
    return im


def add_gambit_logo(ax=None, zorder=3, size=plot_settings["logo_size"]):
    if ax is None:
        ax = plt.gca()
    inset = ax.inset_axes([1.0 - size, 0.0, size, size], zorder=zorder)
    inset.set_aspect("equal", anchor="SW")
    inset.imshow(load_small_gambit_logo(), interpolation='none')
    inset.axis('off')
