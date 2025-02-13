"""
Add annotations to axis
=======================
"""

import importlib.resources

import matplotlib.pyplot as plt
import PIL
import numpy as np


def add_header(header_text, ax=None, pad=2):
    if ax is None:
        ax = plt.gca()
    ax.set_title(
        header_text,
        loc="right",
        pad=pad)


def add_gambit_header(ax=None, version=None, **kwargs):
    if version is not None:
        header_text = f"\\textsf{{GAMBIT {version}}}"
    else:
        header_text = "\\textsf{GAMBIT}"
    add_header(header_text, ax, **kwargs)


def load_small_gambit_logo():
    with importlib.resources.path('gambit_plotting_tools', 'gambit_logo_small.png') as file_name:
        im = PIL.Image.open(file_name)
        return np.array(im)


def add_gambit_logo(ax=None, zorder=3, size=0.27):
    if ax is None:
        ax = plt.gca()
    inset = ax.inset_axes([1.0 - size, 0.0, size, size], zorder=zorder)
    inset.set_aspect("equal", anchor="SW")
    inset.imshow(load_small_gambit_logo(), interpolation='none')
    inset.axis('off')
