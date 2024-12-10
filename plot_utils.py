from copy import deepcopy
from collections import OrderedDict
import os
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import scipy.stats
from scipy.special import gammaincinv

from gambit_colormaps import gambit_std_cmap
import gambit_plot_settings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm16",
    "axes.linewidth": 0.1,
    "axes.edgecolor": "black",
})


def print_confidence_level_table():
    # Adapted from http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html

    # Confidence levels (in number of sigmas) to list in table
    use_sigmas = [
        np.sqrt(scipy.stats.chi2.ppf(0.68, 1)),  
        1.0,
        np.sqrt(scipy.stats.chi2.ppf(0.9, 1)),
        np.sqrt(scipy.stats.chi2.ppf(0.95, 1)),
        2.0,
        np.sqrt(scipy.stats.chi2.ppf(0.99, 1)),
        3.0,
        np.sqrt(scipy.stats.chi2.ppf(0.999, 1)),
        4.0
    ]

    # The corresponding confidence levels, in probabilities
    confidence_levels = [scipy.stats.chi2.cdf(s**2, 1) for s in use_sigmas]

    # The range of degrees of freedom to list in the table
    use_dofs = range(1,3)

    # Print table
    print()
    print("conf. level   " + " ".join([f"{100*ci:>10.5f}%" for ci in confidence_levels]))
    print("#sigmas       " + " ".join([f"{s:>10.5f} " for s in use_sigmas]))
    print("p-value       " + " ".join([f"{(1-ci):>10.5f} " for ci in confidence_levels]))
    print()

    # Print chi^2 values
    for d in use_dofs:
        chi_squared = [ scipy.stats.chi2.ppf( ci, d) for ci in confidence_levels ]
        print(f"X^2  (dof={d})  " + " ".join([f"{c:>10.5f} " for c in chi_squared]))
    print()

    # Print corresponding delta log-likelihoods (assuming Wilks' theorem)
    for d in use_dofs:
        chi_squared = [ scipy.stats.chi2.ppf( ci, d) for ci in confidence_levels ]
        delta_loglike = [-0.5 * c for c in chi_squared]
        print(f"ΔlnL (dof={d})  " + " ".join([f"{dll:>10.5f} " for dll in delta_loglike]))
    print()

    # Print corresponding likelihood ratios (assuming Wilks' theorem)
    for d in use_dofs:
        chi_squared = [ scipy.stats.chi2.ppf( ci, d) for ci in confidence_levels ]
        delta_loglike = [-0.5 * c for c in chi_squared]
        likelihood_ratio = [np.exp(dll) for dll in delta_loglike]
        print(f"L/L' (dof={d})  " + " ".join([f"{llr:>10.5f} " for llr in likelihood_ratio]))
    print()



# Some quick ways to get contour levels for a given list of confidence levels (adapted from pippi)
def get_1D_likelihood_ratio_levels(confidence_levels):
    degrees_of_freedom = 1
    contour_levels = [np.exp(-gammaincinv(0.5 * degrees_of_freedom, conf_level)) for conf_level in confidence_levels]
    return contour_levels

def get_2D_likelihood_ratio_levels(confidence_levels):
    degrees_of_freedom = 2
    contour_levels = [np.exp(-gammaincinv(0.5 * degrees_of_freedom, conf_level)) for conf_level in confidence_levels]
    return contour_levels

def get_1D_delta_loglike_levels(confidence_levels):
    degrees_of_freedom = 1
    contour_levels = [-gammaincinv(0.5 * degrees_of_freedom, conf_level) for conf_level in confidence_levels]
    return contour_levels

def get_2D_delta_loglike_levels(confidence_levels):
    degrees_of_freedom = 2
    contour_levels = [-gammaincinv(0.5 * degrees_of_freedom, conf_level) for conf_level in confidence_levels]
    return contour_levels



def create_folders_if_not_exist(file_path):

    path = Path(file_path)
    
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created the directory: {path.parent}")
    else:
        pass


def read_hdf5_datasets(hdf5_file_and_group_names, requested_datasets, filter_invalid_points=True):

    first_dset_key = requested_datasets[0][0] 

    # If filter_invalid_points = True, we need to also read the bool
    # "_isvalid" dataset corresponding to each requested dataset
    isvalid_keys = []
    if filter_invalid_points:
        temp_requested_datasets = deepcopy(requested_datasets)
        # for key, dset_info in requested_datasets.items():
        for key, dset_info in requested_datasets:
            isvalid_key = key + "_isvalid"
            # temp_requested_datasets[key + "_isvalid"] = (dset_info[0] + "_isvalid", bool)
            temp_requested_datasets.append( (key + "_isvalid", (dset_info[0] + "_isvalid", bool)) )
            isvalid_keys.append(isvalid_key)
        requested_datasets = temp_requested_datasets

    # Initialize data dict
    data = OrderedDict()
    # for key, dset_info in requested_datasets.items():
    for key, dset_info in requested_datasets:
        data[key] = np.array([], dtype=dset_info[1])

    # Loop over files and data sets
    for file_name, group_name in hdf5_file_and_group_names:

        print(f"Reading file: {file_name}")

        f = h5py.File(file_name, "r")
        group = f[group_name]

        # for key, dset_info in requested_datasets.items():
        for key, dset_info in requested_datasets:
            data[key] = np.append(data[key], np.array(group[dset_info[0]], dtype=dset_info[1]))
            print(f"- Read dataset: {dset_info[0]}")

        f.close()

    # Use the "_isvalid" data sets to filter invalid points?
    if filter_invalid_points:

        # Construct the mask
        n_pts = data[first_dset_key].shape[0]
        mask = np.full(n_pts, True)
        for isvalid_key in isvalid_keys:
            mask = np.logical_and(mask, data[isvalid_key])

        # Apply the mask
        for key in data.keys():
            data[key] = data[key][mask]

    # Print dataset length (after filtering)
    n_pts = data[first_dset_key].shape[0]
    print(f"Number of points read: {n_pts}")

    return data


def create_empty_figure_1D(xy_bounds, plot_settings):

    # Get bounds in x and y
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    figwidth = plot_settings["figwidth"]
    figheight = plot_settings["figheight"]
    figheight_figwidth_ratio = figheight / figwidth
    fig = plt.figure(figsize=(figwidth, figheight))

    pad_left = plot_settings["pad_left"]
    pad_right = plot_settings["pad_right"]
    pad_bottom = plot_settings["pad_bottom"]
    pad_top = plot_settings["pad_top"]

    plot_width = 1.0 - pad_left - pad_right
    plot_height = 1.0 - pad_bottom - pad_top

    # Add axes 
    left = pad_left
    bottom = pad_bottom
    ax = fig.add_axes((left, bottom, plot_width, plot_height), frame_on=True)
    ax.set_facecolor(plot_settings["facecolor_plot_1D"])

    # Set frame color and width
    for spine in ax.spines.values():
        spine.set_edgecolor(plot_settings["framecolor_plot_1D"])
        spine.set_linewidth(plot_settings["framewidth_1D"])

    # Axis ranges
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    # Minor ticks
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(plot_settings["n_minor_ticks_per_major_tick"] + 1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(plot_settings["n_minor_ticks_per_major_tick"] + 1))

    # Tick parameters
    major_ticks_color = plot_settings["major_ticks_color_1D"]
    minor_ticks_color = plot_settings["minor_ticks_color_1D"]
    major_ticks_width = plot_settings["major_ticks_width"]
    minor_ticks_width = plot_settings["minor_ticks_width"]
    major_ticks_length = plot_settings["major_ticks_length"]
    minor_ticks_length = plot_settings["minor_ticks_length"]
    major_ticks_pad = plot_settings["major_ticks_pad"]
    minor_ticks_pad = plot_settings["minor_ticks_pad"]

    major_ticks_bottom = plot_settings["major_ticks_bottom"]
    major_ticks_top = plot_settings["major_ticks_top"]
    major_ticks_left = plot_settings["major_ticks_left"]
    major_ticks_right = plot_settings["major_ticks_right"]
    minor_ticks_bottom = plot_settings["minor_ticks_bottom"]
    minor_ticks_top = plot_settings["minor_ticks_top"]
    minor_ticks_left = plot_settings["minor_ticks_left"]
    minor_ticks_right = plot_settings["minor_ticks_right"]

    plt.tick_params(which="major", axis="x",direction="in", color=major_ticks_color, width=major_ticks_width, length=major_ticks_length, pad=major_ticks_pad, bottom=major_ticks_bottom, top=major_ticks_top)
    plt.tick_params(which="minor", axis="x",direction="in", color=minor_ticks_color, width=minor_ticks_width, length=minor_ticks_length,  pad=minor_ticks_pad, bottom=minor_ticks_bottom, top=minor_ticks_top)
    plt.tick_params(which="major", axis="y",direction="in", color=major_ticks_color, width=major_ticks_width, length=major_ticks_length, pad=major_ticks_pad, left=major_ticks_left, right=major_ticks_right)
    plt.tick_params(which="minor", axis="y",direction="in", color=minor_ticks_color, width=minor_ticks_width, length=minor_ticks_length,  pad=minor_ticks_pad, left=minor_ticks_left, right=minor_ticks_right)

    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3,3))

    ax.xaxis.get_offset_text().set_fontsize(plot_settings["fontsize"])
    ax.yaxis.get_offset_text().set_fontsize(plot_settings["fontsize"])

    return fig, ax



def create_empty_figure(xy_bounds, plot_settings):

    # Get bounds in x and y
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    figwidth = plot_settings["figwidth"]
    figheight = plot_settings["figheight"]
    figheight_figwidth_ratio = figheight / figwidth
    fig = plt.figure(figsize=(figwidth, figheight))

    pad_left = plot_settings["pad_left"]
    pad_right = plot_settings["pad_right"]
    pad_bottom = plot_settings["pad_bottom"]
    pad_top = plot_settings["pad_top"]

    plot_width = 1.0 - pad_left - pad_right
    plot_height = 1.0 - pad_bottom - pad_top

    # Add axes 
    left = pad_left
    bottom = pad_bottom
    ax = fig.add_axes((left, bottom, plot_width, plot_height), frame_on=True)
    ax.set_facecolor(plot_settings["facecolor_plot"])

    # Set frame color and width
    for spine in ax.spines.values():
        spine.set_edgecolor(plot_settings["framecolor_plot"])
        spine.set_linewidth(plot_settings["framewidth"])

    # Axis ranges
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    # Minor ticks
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(plot_settings["n_minor_ticks_per_major_tick"] + 1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(plot_settings["n_minor_ticks_per_major_tick"] + 1))

    # Tick parameters
    major_ticks_color = plot_settings["major_ticks_color"]
    minor_ticks_color = plot_settings["minor_ticks_color"]
    major_ticks_width = plot_settings["major_ticks_width"]
    minor_ticks_width = plot_settings["minor_ticks_width"]
    major_ticks_length = plot_settings["major_ticks_length"]
    minor_ticks_length = plot_settings["minor_ticks_length"]
    major_ticks_pad = plot_settings["major_ticks_pad"]
    minor_ticks_pad = plot_settings["minor_ticks_pad"]

    major_ticks_bottom = plot_settings["major_ticks_bottom"]
    major_ticks_top = plot_settings["major_ticks_top"]
    major_ticks_left = plot_settings["major_ticks_left"]
    major_ticks_right = plot_settings["major_ticks_right"]
    minor_ticks_bottom = plot_settings["minor_ticks_bottom"]
    minor_ticks_top = plot_settings["minor_ticks_top"]
    minor_ticks_left = plot_settings["minor_ticks_left"]
    minor_ticks_right = plot_settings["minor_ticks_right"]

    plt.tick_params(which="major", axis="x",direction="in", color=major_ticks_color, width=major_ticks_width, length=major_ticks_length, pad=major_ticks_pad, bottom=major_ticks_bottom, top=major_ticks_top)
    plt.tick_params(which="minor", axis="x",direction="in", color=minor_ticks_color, width=minor_ticks_width, length=minor_ticks_length,  pad=minor_ticks_pad, bottom=minor_ticks_bottom, top=minor_ticks_top)
    plt.tick_params(which="major", axis="y",direction="in", color=major_ticks_color, width=major_ticks_width, length=major_ticks_length, pad=major_ticks_pad, left=major_ticks_left, right=major_ticks_right)
    plt.tick_params(which="minor", axis="y",direction="in", color=minor_ticks_color, width=minor_ticks_width, length=minor_ticks_length,  pad=minor_ticks_pad, left=minor_ticks_left, right=minor_ticks_right)

    plt.ticklabel_format(axis="both", style="sci", scilimits=(-3,3))

    ax.xaxis.get_offset_text().set_fontsize(plot_settings["fontsize"])
    ax.yaxis.get_offset_text().set_fontsize(plot_settings["fontsize"])

    return fig, ax



def bin_and_profile_2d(x_data, y_data, z_data, n_bins, xy_bounds, 
                       already_sorted=False, 
                       z_fill_value=-1*np.finfo(float).max):

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data?
    if not already_sorted:
        # Sort data according to z value, from highest to lowest
        p = np.argsort(z_data)
        p = p[::-1]
        x_data = x_data[p]
        y_data = y_data[p]
        z_data = z_data[p]

    # Get bounds in x and y
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    # Binning of the plane
    n_xbins = n_bins[0]
    n_ybins = n_bins[1]

    x_bin_width = (x_max - x_min) / float(n_xbins)
    y_bin_width = (y_max - y_min) / float(n_ybins)

    x_bin_limits = np.linspace(x_min, x_max, n_xbins + 1)
    y_bin_limits = np.linspace(y_min, y_max, n_ybins + 1)

    x_bin_centres = np.linspace(x_min + 0.5 * x_bin_width, x_max - 0.5 * x_bin_width, n_xbins)
    y_bin_centres = np.linspace(y_min + 0.5 * y_bin_width, y_max - 0.5 * y_bin_width, n_ybins)

    x_bin_indices = np.digitize(x_data, x_bin_limits, right=False) - 1
    y_bin_indices = np.digitize(y_data, y_bin_limits, right=False) - 1

    # Determine the z_value in each bin.
    # 
    # Since we have already sorted all the points from high to low
    # z value, we can just set the z value for each bin using the first 
    # point we encounter that belongs in that given (x,y) bin. All 
    # subsequent points we find that belong in that bin we just skip past 
    # (using the z_val_is_set check below). 

    z_values = np.zeros(n_xbins * n_ybins)
    z_val_is_set = np.array(z_values, dtype=bool)  # Every entry initialised to False

    for i in range(n_pts):

        x_bin_index = x_bin_indices[i]
        y_bin_index = y_bin_indices[i]
        z_values_index = y_bin_index * n_xbins +  x_bin_index

        if (x_bin_index >= n_xbins) or (y_bin_index >= n_ybins):
            continue

        if z_val_is_set[z_values_index]:
            continue

        z_val = z_data[i]
        z_values[z_values_index] = z_val
        z_val_is_set[z_values_index] = True

        if np.all(z_val_is_set):
            break

    # For the (x,y) bins where we don't have any points in plot_data, 
    # we set the z value manually using z_fill_value
    for z_values_index in range(z_values.shape[0]):
            if not z_val_is_set[z_values_index]:
                z_values[z_values_index] = z_fill_value


    # Fill arrays x_values and y_values
    x_values = np.zeros(n_xbins * n_ybins)
    y_values = np.zeros(n_xbins * n_ybins)
    for x_bin_index, x_bin_centre in enumerate(x_bin_centres):
        for y_bin_index, y_bin_centre in enumerate(y_bin_centres):

            point_index = y_bin_index * n_xbins +  x_bin_index
            x_values[point_index] = x_bin_centre
            y_values[point_index] = y_bin_centre

    return x_values, y_values, z_values




def plot_2d_profile(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray, 
                    labels: tuple, n_bins: tuple, xy_bounds = None, 
                    contour_levels = [], z_fill_value = -1*np.finfo(float).max, 
                    z_is_loglike = True, plot_likelihood_ratio = True,
                    add_max_likelihood_marker = True,
                    plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Sanity checks
    if not (x_data.shape == y_data.shape == z_data.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == len(z_data.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data according to z value, from highest to lowest
    p = np.argsort(z_data)
    p = p[::-1]
    x_data = x_data[p]
    y_data = y_data[p]
    z_data = z_data[p]

    # Get the highest and lowest z values
    z_max = z_data[0]
    z_min = z_data[-1]

    # Get plot bounds in x and y
    if xy_bounds is None:
        xy_bounds = ([np.min(x_data), np.max(x_data)], [np.min(y_data), np.max(y_data)])
    xy_bounds[0][0] -= np.finfo(float).eps
    xy_bounds[0][1] += np.finfo(float).eps
    xy_bounds[1][0] -= np.finfo(float).eps
    xy_bounds[1][1] += np.finfo(float).eps
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    # Use loglike difference
    if z_is_loglike:
        z_data = z_data - z_max
        z_max = z_data[0]
        z_min = z_data[-1]

    # Bin and profile data
    x_values, y_values, z_values = bin_and_profile_2d(x_data, y_data, z_data, 
                                                      n_bins, xy_bounds,
                                                      already_sorted=True, 
                                                      z_fill_value=z_fill_value)

    # Convert from lnL - lnL_max = ln(L/Lmax) to L/Lmax 
    if z_is_loglike and plot_likelihood_ratio:
        z_values = np.exp(z_values)

    # Colorbar range
    cmap_vmin = z_min
    cmap_vmax = z_max
    if (z_is_loglike) and (plot_likelihood_ratio):
        cmap_vmin = 0.0
        cmap_vmax = 1.0
    if (z_is_loglike) and (not plot_likelihood_ratio):
        cmap_vmin = z_max - 9.0
        cmap_vmax = z_max

    # Create an empty figure using our plot settings
    fig, ax = create_empty_figure(xy_bounds, plot_settings)

    # Axis labels
    x_label = labels[0]
    y_label = labels[1]

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Create a color scale normalization
    norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)

    # Make color plot of the profile loglike
    n_xbins = n_bins[0]
    n_ybins = n_bins[1]
    x_values = x_values.reshape(n_ybins, n_xbins)
    y_values = y_values.reshape(n_ybins, n_xbins)
    z_values = z_values.reshape(n_ybins, n_xbins)
    im = ax.imshow(z_values, interpolation=plot_settings["interpolation"], aspect="auto", extent=[x_min, x_max, y_min, y_max],
                   cmap=plot_settings["colormap"], norm=norm, origin="lower")

    # Draw contours?
    if len(contour_levels) > 0:
        contour_levels.sort()
        ax.contour(x_values, y_values, z_values, contour_levels, colors=plot_settings["contour_color"], 
                  linewidths=[plot_settings["contour_linewidth"]]*len(contour_levels), linestyles=plot_settings["contour_linestyle"])

    # Add a star at the max-likelihood point
    if (z_is_loglike and add_max_likelihood_marker):
        max_like_index = np.argmax(z_data)
        x_max_like = x_data[max_like_index]
        y_max_like = y_data[max_like_index]
        ax.scatter(x_max_like, y_max_like, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                   edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100)

    # Add a colorbar
    cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"], loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])

    cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
    cbar.outline.set_linewidth(plot_settings["framewidth"])

    cbar.set_ticks(np.linspace(cmap_vmin, cmap_vmax, plot_settings["colorbar_n_major_ticks"]), minor=False)
    cbar.set_ticks(np.linspace(cmap_vmin, cmap_vmax, plot_settings["colorbar_n_minor_ticks"]), minor=True)

    cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
    cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

    cbar_label = labels[2]
    if (z_is_loglike) and (not plot_likelihood_ratio):
        cbar_label = "$\\ln L   - \\ln L_\\mathrm{max}$"
    if (z_is_loglike) and (plot_likelihood_ratio):
        cbar_label = "$\\textrm{Profile likelihood ratio}$ $\\Lambda = L/L_\\mathrm{max}$"
    cbar.set_label(cbar_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax




def plot_1d_posterior(x_data: np.ndarray, posterior_weights: np.ndarray, 
                      x_label: str, n_bins: tuple, x_bounds = None, 
                      credible_regions = [], plot_relative_probability = True, 
                      add_mean_posterior_marker = True,
                      plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Sanity checks
    if not (x_data.shape == posterior_weights.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(posterior_weights.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Get plot bounds in x
    if x_bounds is None:
        x_bounds = (np.min(x_data), np.max(x_data))
    x_bounds[0] -= np.finfo(float).eps
    x_bounds[1] += np.finfo(float).eps
    x_min, x_max = x_bounds

    # Created weighted 1D histogram
    histogram, x_edges = np.histogram(x_data, bins=n_bins, range=x_bounds, weights=posterior_weights)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Set y bound
    y_min = 0.0
    y_max = 1.0
    if not plot_relative_probability:
        y_max = np.max(histogram)

    # Create an empty figure using our plot settings
    xy_bounds = ([x_min, x_max], [y_min, y_max])
    fig, ax = create_empty_figure_1D(xy_bounds, plot_settings)

    # Axis labels
    y_label = "Posterior probability"
    if plot_relative_probability:
        y_label = "Relative probability $P/P_{\\mathrm{max}}$"

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Make a histogram of the 1D posterior distribution
    y_data = histogram
    if plot_relative_probability:
        y_data = y_data / np.max(y_data)

    plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="stepfilled", color=plot_settings["1D_posterior_color"], alpha=plot_settings["1D_posterior_fill_alpha"])
    plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="step", color=plot_settings["1D_posterior_color"])

    # Draw credible region lines?
    line_y_values = []
    if len(credible_regions) > 0:

        # For each requested CR line, find the posterior 
        # density height at which to draw the line. 
        sorted_hist = np.sort(y_data)[::-1]
        cumulative_sum = np.cumsum(sorted_hist)
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        for cr in credible_regions:
            line_y_values.append(sorted_hist[np.searchsorted(normalized_cumulative_sum, cr)])

        line_y_values.sort()

        # Draw the lines
        for line_y_val in line_y_values:
            ax.plot([x_min, x_max], [line_y_val, line_y_val], color="black", linewidth=plot_settings["contour_linewidth"], linestyle="dashed")

    # Add marker at the mean posterior point
    if add_mean_posterior_marker:
        x_mean = np.average(x_data, weights=posterior_weights)
        y_mean = 0.0
        ax.scatter(x_mean, y_mean, marker=plot_settings["posterior_mean_marker"], s=plot_settings["posterior_mean_marker_size"], c=plot_settings["posterior_mean_marker_color"],
                   edgecolor=plot_settings["posterior_mean_marker_edgecolor"], linewidth=plot_settings["posterior_mean_marker_linewidth"], zorder=100, clip_on=False)

    # Return plot
    return fig, ax



def plot_2d_posterior(x_data: np.ndarray, y_data: np.ndarray, posterior_weights: np.ndarray, 
                      labels: tuple, n_bins: tuple, xy_bounds = None, 
                      credible_regions = [], plot_relative_probability = True, 
                      add_mean_posterior_marker = True,
                      plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Sanity checks
    if not (x_data.shape == y_data.shape == posterior_weights.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == len(posterior_weights.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Get plot bounds in x and y
    if xy_bounds is None:
        xy_bounds = ([np.min(x_data), np.max(x_data)], [np.min(y_data), np.max(y_data)])
    xy_bounds[0][0] -= np.finfo(float).eps
    xy_bounds[0][1] += np.finfo(float).eps
    xy_bounds[1][0] -= np.finfo(float).eps
    xy_bounds[1][1] += np.finfo(float).eps
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    # Created weighted 2D histogram
    histogram, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=n_bins, range=xy_bounds, weights=posterior_weights)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Colorbar range
    cmap_vmin = 0.0
    cmap_vmax = np.max(histogram)
    if (plot_relative_probability):
        cmap_vmax = 1.0

    # Create an empty figure using our plot settings
    fig, ax = create_empty_figure(xy_bounds, plot_settings)

    # Axis labels
    x_label = labels[0]
    y_label = labels[1]

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Create a color scale normalization
    norm = matplotlib.cm.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)

    # Make a color plot of the 2D posterior distribution
    X, Y = np.meshgrid(x_centers, y_centers)
    z_data = histogram.T
    if plot_relative_probability:
        z_data = z_data / np.max(z_data)

    im = ax.imshow(z_data, interpolation=plot_settings["interpolation"], aspect="auto", extent=[x_min, x_max, y_min, y_max],
                   cmap=plot_settings["colormap"], norm=norm, origin="lower")

    # Draw credible region contours?
    contour_levels = []
    if len(credible_regions) > 0:

        # For each requested CR contour, find the posterior 
        # density height at which to draw the contour. 
        sorted_hist = np.sort(histogram.ravel())[::-1]
        cumulative_sum = np.cumsum(sorted_hist)
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        for cr in credible_regions:
            contour_levels.append(sorted_hist[np.searchsorted(normalized_cumulative_sum, cr)])

        contour_levels.sort()

        # Draw the contours
        ax.contour(X, Y, histogram.T, contour_levels, colors=plot_settings["contour_color"], 
                   linewidths=[plot_settings["contour_linewidth"]]*len(contour_levels), linestyles=plot_settings["contour_linestyle"])

    # Add marker at the mean posterior point
    if add_mean_posterior_marker:
        x_mean = np.average(x_data, weights=posterior_weights)
        y_mean = np.average(y_data, weights=posterior_weights)
        ax.scatter(x_mean, y_mean, marker=plot_settings["posterior_mean_marker"], s=plot_settings["posterior_mean_marker_size"], c=plot_settings["posterior_mean_marker_color"],
                   edgecolor=plot_settings["posterior_mean_marker_edgecolor"], linewidth=plot_settings["posterior_mean_marker_linewidth"], zorder=100)

    # Add a colorbar
    cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"], loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])

    cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
    cbar.outline.set_linewidth(plot_settings["framewidth"])

    cbar.set_ticks(np.linspace(cmap_vmin, cmap_vmax, plot_settings["colorbar_n_major_ticks"]), minor=False)
    cbar.set_ticks(np.linspace(cmap_vmin, cmap_vmax, plot_settings["colorbar_n_minor_ticks"]), minor=True)

    cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
    cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

    cbar_label = "Posterior probability"
    if (plot_relative_probability):
        cbar_label = "Relative probability $P/P_{\\mathrm{max}}$"
    cbar.set_label(cbar_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax



