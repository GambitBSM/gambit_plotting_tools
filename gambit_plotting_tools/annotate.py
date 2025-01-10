"""
Add annotations to axis
=======================
"""

import matplotlib.pyplot as plt
import PIL
import numpy as np
import urllib.request


def add_header(ax=None, version=None):
    if ax is None:
        ax = plt.gca()
    if version is not None:
        header_text = f"\\textsf{{GAMBIT {version}}}"
    else:
        header_text = "\\textsf{GAMBIT}"
    ax.set_title(header_text, loc="right")


def cropped_gambit_logo():
    url = "https://gambitbsm.org/gambit_logo.png"
    im = PIL.Image.open(urllib.request.urlopen(url))
    im = im.crop((0, 0, int(0.6 * im.size[0]), int(0.5 * im.size[1])))
    im = np.array(im)
    return im


def add_logo(ax=None, zorder=3):
    if ax is None:
        ax = plt.gca()
    inset = ax.inset_axes([0.7, 0., 0.3, 0.3], zorder=zorder)
    inset.set_aspect("equal", anchor="SW")
    inset.imshow(cropped_gambit_logo(), interpolation='none')
    inset.axis('off')
