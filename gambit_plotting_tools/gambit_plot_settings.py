import numpy as np
from gambit_plotting_tools.gambit_colormaps import gambit_std_cmap


plot_settings = {

    "colormap": gambit_std_cmap,

    "interpolation": True,
    "interpolation_resolution": 400,

    "framewidth": 1.2,
    "framecolor_plot": "white",
    "facecolor_plot": "0.5",
    "framecolor_colorbar": "black",

    "framewidth_1D": 1.2,
    "framecolor_plot_1D": "black",
    "facecolor_plot_1D": "white",

    "figwidth": 4.92,  # inches
    "figheight": 4.00, # inches

    "pad_left": 0.16,
    "pad_right": 0.21,
    "pad_bottom": 0.16,
    "pad_top": 0.05,

    "fontsize": 14,

    "1D_posterior_color": "blue",
    "1D_posterior_fill_alpha": 0.2,

    "1D_profile_likelihood_color": "crimson",
    "1D_profile_likelihood_fill_alpha": 0.2,

    "major_ticks_color": "white",
    "minor_ticks_color": "white",
    "major_ticks_color_1D": "black",
    "minor_ticks_color_1D": "black",

    "major_ticks_bottom": True,
    "major_ticks_top": True,
    "major_ticks_left": True,
    "major_ticks_right": True,

    "minor_ticks_bottom": True,
    "minor_ticks_top": True,
    "minor_ticks_left": True,
    "minor_ticks_right": True,

    "major_ticks_width": 0.85,
    "minor_ticks_width": 0.85,
    "major_ticks_length": 10,
    "minor_ticks_length": 5,
    "major_ticks_pad": 5,
    "minor_ticks_pad": 5,

    "n_minor_ticks_per_major_tick": 3,

    "xlabel_pad": 6,
    "ylabel_pad": 12,

    "contour_linewidth": 1.0,
    "contour_color": "white",
    "contour_linestyle": "solid",
    "close_likelihood_contours": True,

    "max_likelihood_marker": "*",
    "max_likelihood_marker_size": 100,
    "max_likelihood_marker_color": "white",
    "max_likelihood_marker_edgecolor": "black",
    "max_likelihood_marker_linewidth": 1.2,

    "posterior_mean_marker": "o",
    "posterior_mean_marker_size": 40,
    "posterior_mean_marker_color": "white",
    "posterior_mean_marker_edgecolor": "black",
    "posterior_mean_marker_linewidth": 1.2,

    "colorbar_width": "6.25%",
    "colorbar_height": "92.5%",
    "colorbar_loc": "right",
    "colorbar_borderpad": -1.87,
    "colorbar_orientation": "vertical",
    "colorbar_label_fontsize": 11,
    "colorbar_label_pad": 18,
    "colorbar_label_rotation": 270,

    "colorbar_minor_ticks_labelsize": 11,
    "colorbar_major_ticks_labelsize": 11,
    "colorbar_minor_ticks_color": "black",
    "colorbar_major_ticks_color": "black",
    "colorbar_minor_ticks_width": 0.85,
    "colorbar_major_ticks_width": 0.85,
    "colorbar_minor_ticks_length": 5,
    "colorbar_major_ticks_length": 10,
    "colorbar_minor_ticks_pad": 2,
    "colorbar_major_ticks_pad": 2,

    "colorbar_n_major_ticks": 6,
    "colorbar_n_minor_ticks": 21,

    "header_fontsize": 7,
    "header_pad": 2,

    "logo_size": 0.27
}

