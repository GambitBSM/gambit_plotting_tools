from copy import copy, deepcopy
from collections import OrderedDict
import os
import shutil
import numpy as np
from itertools import cycle
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy.special import gammaincinv

from gambit_plotting_tools.gambit_colormaps import gambit_std_cmap
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings


# Check if LaTeX rendering is supported by looking for the commands 'latex', 'dvipng' and 'gs'
tex_required_tools = ("latex", "dvipng", "gs")
tex_support_detected = all(shutil.which(cmd) for cmd in tex_required_tools)
if tex_support_detected:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cm16",
    })
else:
    print(f"gambit_plot_utils: Text rendering with LaTeX will be disabled as one of "
            "the tools {tex_required_tools} was not found. Make sure to use Matplotlib's "
            "mathtext syntax for the text labels in your plotting script.")

# Set some axes defaults 
plt.rcParams.update({
    "axes.linewidth": 0.1,
    "axes.edgecolor": "black",
})


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

def get_1D_delta_chi2_levels(confidence_levels):
    degrees_of_freedom = 1
    contour_levels = [2 * gammaincinv(0.5 * degrees_of_freedom, conf_level) for conf_level in confidence_levels]
    return contour_levels

def get_2D_delta_chi2_levels(confidence_levels):
    degrees_of_freedom = 2
    contour_levels = [2 * gammaincinv(0.5 * degrees_of_freedom, conf_level) for conf_level in confidence_levels]
    return contour_levels



def create_folders_if_not_exist(file_path):

    path = Path(file_path)
    
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created the directory: {path.parent}")
    else:
        pass


def collect_all_dataset_names(hdf5_file_and_group_name, leave_out_isvalid_datasets=True):
    dataset_names = []

    file_name, group_name = hdf5_file_and_group_name

    # Open the HDF5 file
    with h5py.File(file_name, 'r') as file:

        def collect_datasets(name):
            if leave_out_isvalid_datasets and name.endswith("_isvalid"):
                pass
            else:
                dataset_names.append(name)
        
        # Visit all datasets in the group and collect the dataset names
        file[group_name].visit(collect_datasets)

    return dataset_names    



def collect_all_model_names(hdf5_file_and_group_name):

    all_dataset_names = collect_all_dataset_names(hdf5_file_and_group_name)

    all_model_names = []
    for dset_name in all_dataset_names:
        if "::primary_parameters::" in dset_name:
            model_name = dset_name.split("@")[1].split("::")[0] 
            if not model_name in all_model_names:
                all_model_names.append(model_name)    

    return all_model_names


def collect_all_model_and_param_names(hdf5_file_and_group_name):

    all_dataset_names = collect_all_dataset_names(hdf5_file_and_group_name)

    all_model_names = collect_all_model_names(hdf5_file_and_group_name)

    model_param_dict = {}
    for model_name in all_model_names:
        model_param_dict[model_name] = []
    for dset_name in all_dataset_names:
        if "::primary_parameters::" in dset_name:
            short_param_name = dset_name.split("::")[-1]
            model_name = dset_name.split("@")[1].split("::")[0] 
            model_param_dict[model_name].append(short_param_name)

    return model_param_dict


def read_hdf5_datasets(hdf5_file_and_group_names, requested_datasets, filter_invalid_points=True, verbose=True):

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

        if verbose:
            print(f"Reading file: {file_name}")

        f = h5py.File(file_name, "r")
        group = f[group_name]

        # for key, dset_info in requested_datasets.items():
        for key, dset_info in requested_datasets:
            data[key] = np.append(data[key], np.array(group[dset_info[0]], dtype=dset_info[1]))
            if verbose:
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
    if verbose:
        print(f"Number of points read: {n_pts}")

    return data



def line_intersection(p1, p2, q1, q2):
    """ Return the intersection of line segment p1 -> p2 with q1 -> q2 or None if there's no intersection """

    p = np.array(p1)
    r = np.array(p2) - np.array(p1)
    q = np.array(q1)
    s = np.array(q2) - np.array(q1)

    cross_rs = np.cross(r, s)
    if cross_rs == 0:
        return None  # Lines are parallel or collinear

    t = np.cross(q - p, s) / cross_rs
    u = np.cross(q - p, r) / cross_rs

    intersection_point = None
    if (0 <= t <= 1) and (0 <= u <= 1):
        intersection_point = p + t * r

    return intersection_point



def get_intersection_points_from_lines(line1, line2):

    # Get the Path objects for each line
    path1 = matplotlib.path.Path(np.column_stack([line1.get_xdata(), line1.get_ydata()]))
    path2 = matplotlib.path.Path(np.column_stack([line2.get_xdata(), line2.get_ydata()]))

    # Vertices of the paths
    vertices1 = path1.vertices
    vertices2 = path2.vertices

    # Find intersection points
    intersection_points_upcrossings = []
    intersection_points_downcrossings = []
    for i in range(len(vertices1) - 1):
        p1 = vertices1[i]
        p2 = vertices1[i + 1]
        for j in range(len(vertices2) - 1):
            q1 = vertices2[j]
            q2 = vertices2[j + 1]
            intersect = line_intersection(p1, p2, q1, q2)
            if intersect is not None:
                if (p1[1] <= q1[1]) and (p2[1] >= q2[1]):
                    intersection_points_upcrossings.append(intersect)
                if (p1[1] >= q1[1]) and (p2[1] <= q2[1]):
                    intersection_points_downcrossings.append(intersect)
    return intersection_points_upcrossings, intersection_points_downcrossings



def save_contour_coordinates(contour, contour_coordinates_output_file, header=""):

    coordinates = []

    # For every contour level
    for path in contour.get_paths():
        # For every closed contour at this level
        for poly in path.to_polygons():
            coordinates.append(poly)
            # Use NaN separator to mark contour breaks
            coordinates.append(np.array([[np.nan, np.nan]]))

    # Concatenate all coordinate arrays into one
    coordinates = np.vstack(coordinates)

    # Save to file
    create_folders_if_not_exist(contour_coordinates_output_file)
    np.savetxt(contour_coordinates_output_file, coordinates, delimiter=',', header=header, comments="")
    print(f"Wrote file: {contour_coordinates_output_file}")



def create_empty_figure_1D(xy_bounds, plot_settings, use_facecolor=None):

    # Get bounds in x and y
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    figwidth = plot_settings["figwidth"]
    figheight = plot_settings["figheight"]
    figheight_figwidth_ratio = figheight / figwidth
    fig = plt.figure(figsize=(figwidth, figheight))

    pad_left = plot_settings["pad_left_1D"]
    pad_right = plot_settings["pad_right_1D"]
    pad_bottom = plot_settings["pad_bottom"]
    pad_top = plot_settings["pad_top"]

    plot_width = 1.0 - pad_left - pad_right
    plot_height = 1.0 - pad_bottom - pad_top

    # Add axes 
    left = pad_left
    bottom = pad_bottom
    ax = fig.add_axes((left, bottom, plot_width, plot_height), frame_on=True)
    if use_facecolor is not None:
        ax.set_facecolor(use_facecolor)
    else:
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



def create_empty_figure_2D(xy_bounds, plot_settings, use_facecolor=None):

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
    if use_facecolor is not None:
        ax.set_facecolor(use_facecolor)
    else:
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



def bin_and_profile_1D(x_data, y_data, n_bins, x_bounds, 
                       already_sorted=False, 
                       y_fill_value=-1*np.finfo(float).max):

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data?
    if not already_sorted:
        # Sort data according to y value, from highest to lowest
        p = np.argsort(y_data)
        p = p[::-1]
        x_data = x_data[p]
        y_data = y_data[p]

    # Get bounds in x and y
    x_min, x_max = x_bounds

    # Binning of the x axis
    # x_bin_width = (x_max - x_min) / float(n_bins)
    # x_bin_limits = np.linspace(x_min, x_max, n_bins + 1)
    # x_bin_centres = np.linspace(x_min + 0.5 * x_bin_width, x_max - 0.5 * x_bin_width, n_bins)
    # x_bin_indices = np.digitize(x_data, x_bin_limits, right=False) - 1

    x_bin_width = (x_max - x_min) / float(n_bins)
    x_bin_limits_inner = np.linspace(x_min + 0.5 * x_bin_width, x_max - 0.5 * x_bin_width, n_bins)
    x_bin_limits = np.array([x_min] + list(x_bin_limits_inner) + [x_max])

    x_bin_centres = np.linspace(x_min, x_max, n_bins + 1)
    x_bin_indices = np.digitize(x_data, x_bin_limits, right=False) - 1

    # Determine the y value in each bin.
    # 
    # Since we have already sorted all the points from high to low
    # y value, we can just set the y value for each bin using the first 
    # point we encounter that belongs in that given x bin. All subsequent 
    # points we find that belong in that bin we just skip past 
    # (using the y_val_is_set check below). 

    y_values = np.zeros(n_bins + 1)
    y_val_is_set = np.array(y_values, dtype=bool)  # Every entry initialised to False

    for i in range(n_pts):

        x_bin_index = x_bin_indices[i]
        y_values_index = x_bin_index

        if (x_bin_index >= n_bins + 1):
            continue

        if y_val_is_set[y_values_index]:
            continue

        y_val = y_data[i]
        y_values[y_values_index] = y_val
        y_val_is_set[y_values_index] = True

        if np.all(y_val_is_set):
            break

    # For the x bins where we don't have any points in plot_data, 
    # we set the y value manually using y_fill_value
    for y_values_index in range(y_values.shape[0]):
            if not y_val_is_set[y_values_index]:
                y_values[y_values_index] = y_fill_value

    # # Fill array x_values
    # x_values = np.zeros(n_bins)
    # for x_bin_index, x_bin_centre in enumerate(x_bin_centres):
    #     x_values[x_bin_index] = x_bin_centre

    return x_bin_centres, y_values



def bin_and_profile_2D(x_data, y_data, z_data, n_bins, xy_bounds, 
                       c_data=None, already_sorted=False, 
                       z_fill_value=np.nan, c_fill_value=np.nan):

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    z_data = np.copy(z_data)
    c_data = np.copy(c_data) if c_data is not None else None

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
        if c_data is not None:
            c_data = c_data[p]

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


    # Determine the z value in each bin.
    # 
    # Since we have already sorted all the points from high to low
    # z value, we can just set the z value for each bin using the first 
    # point we encounter that belongs in that given (x,y) bin. All 
    # subsequent points we find that belong in that bin we just skip past 
    # (using the z_val_is_set check below). 

    z_values = np.full(n_xbins * n_ybins, z_fill_value)
    # z_values = np.zeros(n_xbins * n_ybins)
    if c_data is None:
        c_values = np.full(n_xbins * n_ybins, z_fill_value)
    else:
        c_values = np.full(n_xbins * n_ybins, c_fill_value)
    
    z_val_is_set = np.zeros(n_xbins * n_ybins, dtype=bool)  # Every entry initialised to False

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
        if c_data is not None:
            c_val = c_data[i]
            c_values[z_values_index] = c_val
        else:
            c_values[z_values_index] = z_val
        z_val_is_set[z_values_index] = True

        if np.all(z_val_is_set):
            break

    # For the (x,y) bins where we don't have any points in plot_data, 
    # z_values is already filled with z_fill_value.
    # For c_values, if c_data was provided, those bins are already c_fill_value.
    # If c_data was not provided, those bins should be z_fill_value (already done during initialization).
    # So, no explicit loop is needed here if initialization is correct.

    # Fill arrays x_values and y_values
    x_values = np.zeros(n_xbins * n_ybins)
    y_values = np.zeros(n_xbins * n_ybins)
    for x_bin_index, x_bin_centre in enumerate(x_bin_centres):
        for y_bin_index, y_bin_centre in enumerate(y_bin_centres):

            point_index = y_bin_index * n_xbins +  x_bin_index
            x_values[point_index] = x_bin_centre
            y_values[point_index] = y_bin_centre

    return x_values, y_values, z_values, c_values




def plot_1D_profile(x_data: np.ndarray, y_data: np.ndarray, 
                    x_label: str, n_bins: tuple, x_bounds = None, 
                    confidence_levels = [], y_fill_value = -1*np.finfo(float).max, 
                    y_is_loglike = True, plot_likelihood_ratio = True, reverse_sort = False,
                    add_max_likelihood_marker = True, fill_color_below_graph = True, 
                    shaded_confidence_interval_bands=True,
                    plot_settings = gambit_plot_settings.plot_settings,
                    return_plot_details = False,
                    graph_coordinates_output_file=None) -> None:

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    x_bounds = deepcopy(x_bounds) if x_bounds is not None else None

    # Sanity checks
    if not (x_data.shape == y_data.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data according to y value, from highest to lowest
    if reverse_sort:
        p = np.argsort(-1.0 * y_data)
    else:
        p = np.argsort(y_data)
    p = p[::-1]
    x_data = x_data[p]
    y_data = y_data[p]

    # Get the highest and lowest z values
    y_max = y_data[0]
    y_min = y_data[-1]

    # Get plot bounds in x
    if x_bounds is None:
        x_bounds = (np.min(x_data), np.max(x_data))
    x_bounds[0] -= np.finfo(float).eps
    x_bounds[1] += np.finfo(float).eps
    x_min, x_max = x_bounds

    # Use loglike difference
    if y_is_loglike:
        y_data = y_data - y_max
        y_max = y_data[0]
        y_min = y_data[-1]

    # Bin and profile data
    x_values, y_values = bin_and_profile_1D(x_data, y_data, 
                                            n_bins, x_bounds,
                                            already_sorted=True, 
                                            y_fill_value=y_fill_value)

    # Convert from lnL - lnL_max = ln(L/Lmax) to L/Lmax 
    if y_is_loglike and plot_likelihood_ratio:
        y_values = np.exp(y_values)

    if graph_coordinates_output_file is not None:
        np.savetxt(graph_coordinates_output_file, np.column_stack((x_values, y_values)), delimiter=',', header="#x, y", comments="")
        print(f"Wrote file: {graph_coordinates_output_file}")

    # Set y bound
    y_min = np.min(y_values)
    y_max = np.max(y_values)
    if (y_is_loglike) and (not plot_likelihood_ratio):
        y_min = 0.0
        y_max = 6.0
    if (y_is_loglike) and (plot_likelihood_ratio):
        y_min = 0.0
        y_max = 1.0

    # Create an empty figure using our plot settings
    xy_bounds = ([x_min, x_max], [y_min, y_max])
    fig, ax = create_empty_figure_1D(xy_bounds, plot_settings)

    # Axis labels
    y_label = "Likelihood"
    if (y_is_loglike) and (not plot_likelihood_ratio):
        y_label = r"$\ln L   - \ln L_{\mathrm{max}}$"
    if (y_is_loglike) and (plot_likelihood_ratio):
        y_label = r"Profile likelihood ratio $\Lambda = L/L_{\mathrm{max}}$"

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Determine confidence level lines
    cl_lines_y_vals = []
    if len(confidence_levels) > 0:
        cl_lines_y_vals = []
        if (y_is_loglike) and (not plot_likelihood_ratio):
            cl_lines_y_vals = get_1D_delta_loglike_levels(confidence_levels)
        if (y_is_loglike) and (plot_likelihood_ratio):
            cl_lines_y_vals = get_1D_likelihood_ratio_levels(confidence_levels)

    plot_details = {}
    if return_plot_details:
        plot_details["cl_lines_y_vals"] = cl_lines_y_vals

    # Make a 1D profile likelihood plot

    # Fill?
    if fill_color_below_graph:
        plt.fill_between(
                x=x_values, 
                y1=y_values, 
                color=plot_settings["1D_profile_likelihood_color"],
                alpha=plot_settings["1D_profile_likelihood_fill_alpha"],
                linewidth=0.0)

    main_graph, = plt.plot(x_values, y_values, linestyle="solid", color=plot_settings["1D_profile_likelihood_color"])

    if return_plot_details:
        plot_details["main_graph"] = main_graph

    # Add shaded confidence interval bands?
    if shaded_confidence_interval_bands:

        cl_fill_between_coordinates = []

        for cl_line_y_val in cl_lines_y_vals:

            cl_line = matplotlib.lines.Line2D([x_min, x_max], [cl_line_y_val, cl_line_y_val])

            ip_up, ip_dn = get_intersection_points_from_lines(main_graph, cl_line)

            fill_starts_x = [ip[0] for ip in ip_up]
            fill_ends_x = [ip[0] for ip in ip_dn]

            fill_starts_y = [ip[1] for ip in ip_up]
            fill_ends_y = [ip[1] for ip in ip_dn]

            if y_values[0] > cl_line_y_val:
                fill_starts_x = [x_min] + fill_starts_x 
                fill_starts_y = [y_values[0]] + fill_starts_y 

            if len(fill_starts_x) == len(fill_ends_x) + 1:
                fill_ends_x = fill_ends_x + [x_max]
                fill_ends_y = fill_ends_y + [y_values[-1]]

            if len(fill_ends_x) == len(fill_starts_x) + 1:
                fill_starts_x = [x_min] + fill_starts_x
                fill_starts_y = [y_values[0]] + fill_starts_y

            if len(fill_starts_x) != len(fill_ends_x):
                raise Exception("The lists fill_starts_x and fill_ends_x have different lengths. This should not happen.")

            cl_fill_between_coordinates.append({
                "fill_starts_x": copy(fill_starts_x),
                "fill_ends_x": copy(fill_ends_x),
                "fill_starts_y": copy(fill_starts_y),
                "fill_ends_y": copy(fill_ends_y),
            })

        plot_details["cl_fill_between_coordinates"] = cl_fill_between_coordinates

        for i in range(len(cl_lines_y_vals)):

            fill_starts_x = cl_fill_between_coordinates[i]["fill_starts_x"]
            fill_ends_x = cl_fill_between_coordinates[i]["fill_ends_x"]
            fill_starts_y = cl_fill_between_coordinates[i]["fill_starts_y"]
            fill_ends_y = cl_fill_between_coordinates[i]["fill_ends_y"]

            for j in range(len(fill_starts_x)):

                x_start, x_end = fill_starts_x[j], fill_ends_x[j]
                y_start, y_end = fill_starts_y[j], fill_ends_y[j]

                use_x_values = np.array([x_start] + list(x_values[(x_values > x_start) & (x_values < x_end)]) + [x_end])
                use_y_values = np.array([y_start] + list(y_values[(x_values > x_start) & (x_values < x_end)]) + [y_end])

                plt.fill_between(
                        x=use_x_values, 
                        y1=use_y_values,
                        y2=y_min,
                        color=plot_settings["1D_profile_likelihood_color"],
                        alpha=plot_settings["1D_profile_likelihood_fill_alpha"],
                    linewidth=0.0)

    # Draw confidence level lines
    if len(confidence_levels) > 0:

        linewidths = cycle(list(plot_settings["contour_linewidths"]))

        for i,cl in enumerate(confidence_levels):
            cl_line_y_val = cl_lines_y_vals[i]
            ax.plot([x_min, x_max], [cl_line_y_val, cl_line_y_val], color=plot_settings["1D_profile_likelihood_color"], linewidth=next(linewidths), linestyle="dashed")
            cl_text = f"${100*cl:.1f}\\%\\,$CL"
            ax.text(0.06, cl_line_y_val, cl_text, ha="left", va="bottom", fontsize=plot_settings["header_fontsize"], 
                    color=plot_settings["1D_profile_likelihood_color"], transform = ax.transAxes)

    # Add a star at the max-likelihood point
    if (y_is_loglike and add_max_likelihood_marker):
        max_like_index = np.argmax(y_data)
        x_max_like = x_data[max_like_index]
        ax.scatter(x_max_like, 0.0, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                   edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100, clip_on=False)
        plot_details["max_like_coordinate"] = x_max_like

    # Return plot
    if return_plot_details:
        return fig, ax, plot_details
    else:
        return fig, ax


def plot_1D_delta_chi2(x_data: np.ndarray, y_data: np.ndarray,
                       x_label: str, n_bins: tuple, x_bounds = None,
                       confidence_levels = [], y_fill_value = np.finfo(float).max,
                       y_data_is_chi2 = True, reverse_sort = False,
                       add_best_fit_marker = True, fill_color_below_graph = True,
                       shaded_confidence_interval_bands=True,
                       plot_settings = gambit_plot_settings.plot_settings,
                       return_plot_details = False,
                       graph_coordinates_output_file=None) -> None:

    from scipy.interpolate import interp1d

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    x_bounds = deepcopy(x_bounds) if x_bounds is not None else None

    # Sanity checks
    if not (x_data.shape == y_data.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Convert from log-likelihood to chi^2 if needed
    if not y_data_is_chi2:
        y_data = -2.0 * y_data

    # Sort data according to y value, from lowest to highest (lowest chi^2 = best fit)
    if reverse_sort:
        p = np.argsort(-1.0 * y_data)
    else:
        p = np.argsort(y_data)
    x_data = x_data[p]
    y_data = y_data[p]

    # Get the lowest and highest chi^2 values
    y_min = y_data[0]
    y_max = y_data[-1]

    # Get plot bounds in x
    if x_bounds is None:
        x_bounds = (np.min(x_data), np.max(x_data))
    x_bounds[0] -= np.finfo(float).eps
    x_bounds[1] += np.finfo(float).eps
    x_min, x_max = x_bounds

    # Use delta chi^2 (relative to minimum)
    y_data = y_data - y_min
    y_max = y_data[-1]
    y_min = y_data[0]  # Should be 0.0

    # Bin and profile data (find minimum chi^2 in each bin)
    # Since we sorted from lowest to highest chi^2, bin_and_profile_1D will
    # automatically pick the minimum (first encountered) value in each bin
    x_values, y_values = bin_and_profile_1D(x_data, y_data,
                                            n_bins, x_bounds,
                                            already_sorted=True,
                                            y_fill_value=y_fill_value)

    if graph_coordinates_output_file is not None:
        np.savetxt(graph_coordinates_output_file, np.column_stack((x_values, y_values)), delimiter=',', header="#x, y", comments="")
        print(f"Wrote file: {graph_coordinates_output_file}")

    # Set y bound
    y_min = np.min(y_values)
    y_max = np.max(y_values)
    # Set reasonable upper limit for delta chi^2 plot
    if y_max < 10.0:
        y_max = 10.0
    else:
        y_max = min(y_max * 1.1, 20.0)  # Cap at 20
    y_min = 0.0

    # Create an empty figure using our plot settings
    xy_bounds = ([x_min, x_max], [y_min, y_max])
    fig, ax = create_empty_figure_1D(xy_bounds, plot_settings)

    # Axis labels
    y_label = r"$\Delta\chi^2$"

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Determine confidence level lines
    cl_lines_y_vals = []
    if len(confidence_levels) > 0:
        cl_lines_y_vals = get_1D_delta_chi2_levels(confidence_levels)

    plot_details = {}
    if return_plot_details:
        plot_details["cl_lines_y_vals"] = cl_lines_y_vals

    # Make a 1D delta chi^2 plot

    # High-res interpolation to get the correct coloring with fill_between
    y_interp = interp1d(x_values, y_values, kind='linear')
    x_values_interp = np.linspace(x_min, x_max, max(x_values.shape[0], 1000))
    y_values_interp = y_interp(x_values_interp)

    # Fill?
    if fill_color_below_graph:
        plt.fill_between(
                x=x_values_interp,
                y1=y_values_interp,
                color=plot_settings["1D_profile_likelihood_color"],
                alpha=plot_settings["1D_profile_likelihood_fill_alpha"],
                linewidth=0.0)

    # Plot main graph
    main_graph, = plt.plot(x_values, y_values, linestyle="solid", color=plot_settings["1D_profile_likelihood_color"])

    if return_plot_details:
        plot_details["main_graph"] = main_graph

    # Add shaded confidence interval bands?
    if shaded_confidence_interval_bands:
        for cl_line_y_val in cl_lines_y_vals:
            plt.fill_between(
                    x=x_values_interp,
                    y1=y_values_interp,
                    y2=y_min,
                    where=(y_values_interp < cl_line_y_val),
                    color=plot_settings["1D_profile_likelihood_color"],
                    alpha=plot_settings["1D_profile_likelihood_fill_alpha"],
                    interpolate=True,
                    linewidth=0.0)

    # Draw confidence level lines
    if len(confidence_levels) > 0:

        linewidths = cycle(list(plot_settings["contour_linewidths"]))

        for i,cl in enumerate(confidence_levels):
            cl_line_y_val = cl_lines_y_vals[i]
            ax.plot([x_min, x_max], [cl_line_y_val, cl_line_y_val], color=plot_settings["1D_profile_likelihood_color"], linewidth=next(linewidths), linestyle="dashed")
            cl_text = f"${100*cl:.1f}\\%\\,$CL"
            ax.text(x_min + 0.06*(x_max - x_min), cl_line_y_val, cl_text, ha="left", va="bottom", fontsize=plot_settings["header_fontsize"],
                    color=plot_settings["1D_profile_likelihood_color"])

    # Add a star at the best-fit point (delta chi^2 = 0)
    if add_best_fit_marker:
        min_chi2_index = np.argmin(y_data)
        x_best_fit = x_data[min_chi2_index]
        ax.scatter(x_best_fit, 0.0, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                   edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100, clip_on=False)
        plot_details["best_fit_coordinate"] = x_best_fit

    # Return plot
    if return_plot_details:
        return fig, ax, plot_details
    else:
        return fig, ax


def plot_2D_profile(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray, 
                    labels: tuple, n_bins: tuple, xy_bounds = None, z_bounds = None,
                    contour_levels = [], contour_coordinates_output_file = None,
                    z_is_loglike = True, plot_likelihood_ratio = True, reverse_sort = False,
                    add_max_likelihood_marker = True,
                    color_data: np.ndarray = None, color_label: str = None, 
                    color_bounds = None, color_z_condition = None,
                    missing_value_color= None,
                    plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    z_data = np.copy(z_data)
    color_data = np.copy(color_data) if color_data is not None else None
    xy_bounds = deepcopy(xy_bounds) if xy_bounds is not None else None
    contour_levels = list(contour_levels) if contour_levels is not None else []

    # Sanity checks
    if not (x_data.shape == y_data.shape == z_data.shape):
        raise Exception("Input arrays x_data, y_data, z_data must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == len(z_data.shape) == 1):
        raise Exception("Input arrays x_data, y_data, z_data must be one-dimensional.")

    if color_data is not None:
        if not (color_data.shape == x_data.shape):
            raise Exception("Input array color_data must have the same shape as x_data, y_data, z_data.")
        if not (len(color_data.shape) == 1):
            raise Exception("Input array color_data must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data according to z value, from highest to lowest
    if reverse_sort:
        p = np.argsort(-1.0 * z_data)
    else:
        p = np.argsort(z_data)
    p = p[::-1]
    x_data = x_data[p]
    y_data = y_data[p]
    z_data = z_data[p]
    if color_data is None:
        c_data = z_data
    else:
        c_data = color_data[p]

    # Get the highest and lowest z values
    z_max = z_data[0]
    z_min = z_data[-1]

    # Get plot bounds in x and y
    if xy_bounds is None:
        xy_bounds = ([np.min(x_data), np.max(x_data)], [np.min(y_data), np.max(y_data)]) # Use sorted data for bounds if not provided
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
        if color_data is None:
            c_data = z_data

    # Fill values
    z_fill_value = np.nan
    c_fill_value = np.nan

    # Bin and profile data.
    # Pass sorted data (x_data, y_data, z_data, c_data) and already_sorted=True
    x_values, y_values, z_values, c_values = bin_and_profile_2D(
        x_data, y_data, z_data, n_bins, xy_bounds,
        c_data=c_data, 
        already_sorted=True, 
        z_fill_value=z_fill_value, 
        c_fill_value=c_fill_value
    )

    # Convert from lnL - lnL_max = ln(L/Lmax) to L/Lmax 
    if z_is_loglike and plot_likelihood_ratio:
        z_values = np.exp(z_values)
        if color_data is None:  # c_data = z_data
            c_values = np.exp(c_values)

    # Max and min values for the color dataset
    c_max = np.nanmax(c_values)
    c_min = np.nanmin(c_values)

    # Colorbar range
    if color_data is None:   # c_data = z_data
        if z_bounds is None:
            if z_is_loglike and plot_likelihood_ratio:
                color_plot_bounds = (0.0, 1.0)
            elif z_is_loglike and not plot_likelihood_ratio:
                color_plot_bounds = (z_max - 9.0, z_max)
            else:
                color_plot_bounds = (z_min, z_max)
        else:
            color_plot_bounds = z_bounds
    else:
        if color_bounds is None:
            color_plot_bounds = (c_min, c_max)
        else:
            color_plot_bounds = color_bounds
            
    if missing_value_color is None:
        missing_value_color = plot_settings["facecolor_plot"]
    fig, ax = create_empty_figure_2D(xy_bounds, plot_settings, use_facecolor=missing_value_color)

    # Axis labels
    x_label = labels[0]
    y_label = labels[1]

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Create a color scale normalization using the determined bounds
    norm = matplotlib.cm.colors.Normalize(vmin=color_plot_bounds[0], vmax=color_plot_bounds[1])

    # Reshape data for plotting
    n_xbins = n_bins[0]
    n_ybins = n_bins[1]
    x_values = x_values.reshape(n_ybins, n_xbins)
    y_values = y_values.reshape(n_ybins, n_xbins)
    z_values = z_values.reshape(n_ybins, n_xbins)
    c_values = c_values.reshape(n_ybins, n_xbins)

    # Do interpolation?
    x_values_plot = x_values
    y_values_plot = y_values
    z_values_plot = z_values
    c_values_plot = c_values
    if plot_settings["interpolation"]:
        x_values_interpolated, y_values_interpolated, c_values_interpolated = grid_2D_interpolation(x_values, y_values, c_values, plot_settings["interpolation_resolution"])
        _, _, z_values_interpolated = grid_2D_interpolation(x_values, y_values, z_values, plot_settings["interpolation_resolution"])

        if z_is_loglike and plot_settings["close_likelihood_contours"]:
            z_values_interpolated[np.isnan(z_values_interpolated)] = np.nanmin(z_values_interpolated)

        x_values_plot = x_values_interpolated
        y_values_plot = y_values_interpolated
        z_values_plot = z_values_interpolated
        c_values_plot = c_values_interpolated.T  # Transpose is due to how imshow works
    else:
        if z_is_loglike and plot_settings["close_likelihood_contours"]:
            z_values_plot[np.isnan(z_values_plot)] = np.nanmin(z_values_plot)

    # Mask some part of the colored data?
    if (color_data is not None) and (color_z_condition is not None):
        if plot_settings["interpolation"]:
            mask = np.logical_not(color_z_condition(z_values_plot.T))
        else:
            mask = np.logical_not(color_z_condition(z_values_plot))
        c_values_plot[mask] = np.nan

    # Make color plot
    im = ax.imshow(c_values_plot, aspect="auto", extent=[x_min, x_max, y_min, y_max],
                   cmap=plot_settings["colormap"], norm=norm, origin="lower")

    # Draw contours?
    if len(contour_levels) > 0:
        contour_levels.sort()
        contour = ax.contour(x_values_plot, y_values_plot, z_values_plot, contour_levels, 
                             colors=list(plot_settings["contour_colors"]), 
                             linewidths=list(plot_settings["contour_linewidths"]),
                             linestyles=list(plot_settings["contour_linestyles"]))

        # Save contour coordinates to file?
        if contour_coordinates_output_file != None:
            header = "# x,y coordinates for profile likelihood contours at the likelihood ratio values " + ", ".join([f"{l:.4e}" for l in contour_levels]) + ". Sets of coordinates for individual closed contours are separated by nan entries."
            save_contour_coordinates(contour, contour_coordinates_output_file, header=header)

    # Add a star at the max-likelihood point
    if (z_is_loglike and add_max_likelihood_marker):
        x_max_like = x_data[0] # x_data has already been sorted according to z_data
        y_max_like = y_data[0] # y_data has already been sorted according to z_data
        ax.scatter(x_max_like, y_max_like, marker=plot_settings["max_likelihood_marker"], 
                   s=plot_settings["max_likelihood_marker_size"], 
                   c=plot_settings["max_likelihood_marker_color"],
                   edgecolor=plot_settings["max_likelihood_marker_edgecolor"], 
                   linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100)

    # Add a colorbar
    cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"], 
                         loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])

    cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
    cbar.outline.set_linewidth(plot_settings["framewidth"])

    cbar.set_ticks(np.linspace(color_plot_bounds[0], color_plot_bounds[1], plot_settings["colorbar_n_major_ticks"]), minor=False)
    minor_tick_values = np.linspace(color_plot_bounds[0], color_plot_bounds[1], plot_settings["colorbar_n_minor_ticks"])
    cbar.set_ticks(minor_tick_values[(minor_tick_values >= color_plot_bounds[0]) & (minor_tick_values <= color_plot_bounds[1])], minor=True)

    cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
    cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

    # Determine colorbar label
    cbar_label = ""
    if color_data is None:
        cbar_label = labels[2] # Default z-axis label
        if (z_is_loglike) and (not plot_likelihood_ratio):
            cbar_label = r"$\ln L   - \ln L_{\mathrm{max}}$"
        if (z_is_loglike) and (plot_likelihood_ratio):
            cbar_label = r"Profile likelihood ratio $\Lambda = L/L_{\mathrm{max}}$"
    else:
        if color_label is None:
            cbar_label = "[Missing label]"
        else:
            cbar_label = color_label
            
    cbar.set_label(cbar_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax



def plot_conditional_profile_intervals(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray, 
                                       labels: tuple, n_bins: tuple, xy_bounds=None, 
                                       z_bounds=None, confidence_levels=[],
                                       draw_interval_limits=True,
                                       draw_interval_connectors=True,
                                       add_max_likelihood_marker=True,
                                       shaded_confidence_interval_bands=False,
                                       x_condition="bin",
                                       missing_value_color = None, 
                                       plot_settings=gambit_plot_settings.plot_settings):

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    z_data = np.copy(z_data)
    xy_bounds = deepcopy(xy_bounds) if xy_bounds is not None else None

    # This function only works for profile likelihood ratio plots
    z_is_loglike = True
    plot_likelihood_ratio = True

    known_x_conditions = ["bin", "upperbound", "lowerbound"]
    if x_condition not in known_x_conditions:
        raise Exception(f"Argument 'x_condition' must be one of {', '.join(known_x_conditions)}.")

    # Sort data according to z value, from highest to lowest
    p = np.argsort(z_data)
    p = p[::-1]
    x_data = x_data[p]
    y_data = y_data[p]
    z_data = z_data[p]

    # Since z is loglike, adjust data
    z_data = z_data - np.max(z_data)

    # Get the highest and lowest z values
    z_max = z_data[0]
    z_min = z_data[-1]

    # Determine bounds
    if xy_bounds is None:
        # Determine bounds from data if not provided
        x_min_data, x_max_data = np.min(x_data), np.max(x_data)
        y_min_data, y_max_data = np.min(y_data), np.max(y_data)
        xy_bounds = ([x_min_data, x_max_data], [y_min_data, y_max_data])

    xy_bounds[0][0] -= np.finfo(float).eps
    xy_bounds[0][1] += np.finfo(float).eps
    xy_bounds[1][0] -= np.finfo(float).eps
    xy_bounds[1][1] += np.finfo(float).eps
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    if z_bounds is None:
        z_bounds = (0.0, 1.0)

    # Define x-bins
    x_bin_limits = np.linspace(x_min, x_max, n_bins[0] + 1)

    # Loop over x bins
    Z_values_2D = np.full((n_bins[1] + 1, n_bins[0]), np.nan)
    if plot_settings["interpolation"]:
        Z_values_2D = np.full((plot_settings["interpolation_resolution"] + 1, n_bins[0]), np.nan)
        y_coarse = np.linspace(y_min, y_max, n_bins[1] + 1)
        y_fine = np.linspace(y_min, y_max, plot_settings["interpolation_resolution"] + 1)

    ci_boundaries = []
    max_likelihood_coordinates = []

    for i in range(n_bins[0]):
        # Select data for current x-bin
        current_x_min = x_bin_limits[i]
        current_x_max = x_bin_limits[i+1]

        if x_condition == "bin":
            if i == n_bins[0] - 1: # Handle the right edge of the last bin
                mask = (x_data >= current_x_min) & (x_data <= current_x_max)
            else:
                mask = (x_data >= current_x_min) & (x_data < current_x_max)

        elif x_condition == "upperbound":
            mask = (x_data <= current_x_max)

        elif x_condition == "lowerbound":
            mask = (x_data >= current_x_min)

        # x_subset = x_data[mask]
        y_subset = deepcopy(y_data[mask])
        z_subset = deepcopy(z_data[mask])

        # If no data, continue to next x bin
        if len(y_subset) == 0:
            if add_max_likelihood_marker:
                 max_likelihood_coordinates.append(np.nan)
            ci_boundaries.append([])
            continue

        _, _, plot_details = plot_1D_profile(
            y_subset, z_subset, labels[1], n_bins[1], xy_bounds[1], 
            confidence_levels = confidence_levels, 
            y_fill_value = -1*np.finfo(float).max,
            y_is_loglike = True, 
            plot_likelihood_ratio = True,
            add_max_likelihood_marker = add_max_likelihood_marker, 
            fill_color_below_graph = True, 
            shaded_confidence_interval_bands=True,
            plot_settings = plot_settings,
            return_plot_details = True
        )
        plt.close()  # plot_1D_profile creates new figures, so we should close them 

        # Get the Z data
        use_z = plot_details["main_graph"].get_ydata()    
        if plot_settings["interpolation"]:
            use_z = np.interp(y_fine, y_coarse, use_z)
        Z_values_2D[:, i] = use_z

        # Get confidence interval boundaries
        ci_boundaries.append(plot_details["cl_fill_between_coordinates"])

        # Get marker position
        if add_max_likelihood_marker:
            if "max_like_coordinate" in plot_details:
                max_likelihood_coordinates.append(plot_details["max_like_coordinate"])
            else:
                max_likelihood_coordinates.append(np.nan)

    # Create an empty figure using our plot settings
    if shaded_confidence_interval_bands:
        fig, ax = create_empty_figure_1D(xy_bounds, plot_settings)
    else:
        if missing_value_color is None:
            missing_value_color = plot_settings["facecolor_plot"]
        fig, ax = create_empty_figure_2D(xy_bounds, plot_settings, use_facecolor=missing_value_color)


    # Plot 2D color plot?
    if not shaded_confidence_interval_bands:

        # Create color normalization
        norm = matplotlib.cm.colors.Normalize(vmin=z_bounds[0], vmax=z_bounds[1])

        im = ax.imshow(
            Z_values_2D,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            aspect="auto",
            cmap=plot_settings["colormap"],
            interpolation="none",
            norm=norm,
        )

    # Loop over x bins to draw intervals, markers, etc.
    for i in range(n_bins[0]):
        current_x_min = x_bin_limits[i]
        current_x_max = x_bin_limits[i+1]
        current_x_bin_centre = 0.5 * (current_x_min + current_x_max)

        x_bin_ci_bounds = ci_boundaries[i]

        for ci_idx, ci_dict in enumerate(x_bin_ci_bounds):

            if draw_interval_limits:
                use_color = plot_settings["contour_colors"][ci_idx % len(plot_settings["contour_colors"])]
                use_linewidth = plot_settings["contour_linewidths"][ci_idx % len(plot_settings["contour_linewidths"])]
                use_linestyle = plot_settings["contour_linestyles"][ci_idx % len(plot_settings["contour_linestyles"])]

            ci_starts = ci_dict["fill_starts_x"]
            ci_ends = ci_dict["fill_ends_x"]

            assert(len(ci_starts) == len(ci_ends))
            for start,end in zip(ci_starts, ci_ends):

                if shaded_confidence_interval_bands:
                    plt.fill_between(
                        x=[current_x_min, current_x_max],
                        y1=[start, start], 
                        y2=[end, end], 
                        color=plot_settings["1D_profile_likelihood_color"],
                        alpha=plot_settings["1D_profile_likelihood_fill_alpha"],
                        linewidth=0.0,
                        zorder=0,
                    )

                if draw_interval_limits:
                    plt.plot(
                        [current_x_min, current_x_max], [start, start], 
                        color=use_color, 
                        linewidth=use_linewidth,
                        linestyle=use_linestyle,
                        zorder=0,
                    )
                    plt.plot(
                        [current_x_min, current_x_max], [end, end], 
                        color=use_color, 
                        linewidth=use_linewidth,
                        linestyle=use_linestyle,
                        zorder=0,
                    )
                if draw_interval_connectors:
                    plt.plot(
                        [current_x_bin_centre, current_x_bin_centre], [start, end], 
                        color=plot_settings["connector_color"], 
                        linewidth=plot_settings["connector_linewidth"],
                        linestyle=plot_settings["connector_linestyle"],
                        zorder=0,
                    )

        if add_max_likelihood_marker and not np.isnan(max_likelihood_coordinates[i]):
            max_like_y_coord = max_likelihood_coordinates[i]
            ax.scatter(current_x_bin_centre, max_like_y_coord, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                       edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100, clip_on=False)

        # Add vertical separators
        if i > 0:
            plt.plot(
                [current_x_min, current_x_min], [y_min, y_max], 
                color=plot_settings["separator_color"],
                linewidth=plot_settings["separator_linewidth"],
                linestyle="solid",
                zorder=1,
            )

    # Set axis labels
    x_label = labels[0]
    if x_condition == "upperbound":
        x_label = f"{x_label}, upper bound"
    elif x_condition == "lowerbound":
        x_label = f"{x_label}, lower bound"
    y_label = labels[1]
    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Add colorbar
    if shaded_confidence_interval_bands:
        cbar_ax = None
    else:
        cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"],
                             loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])
        cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
        cbar.outline.set_linewidth(plot_settings["framewidth"])

        cbar.set_ticks(np.linspace(z_bounds[0], z_bounds[1], plot_settings["colorbar_n_major_ticks"]), minor=False)
        minor_tick_values = np.linspace(z_bounds[0], z_bounds[1], plot_settings["colorbar_n_minor_ticks"])
        cbar.set_ticks(minor_tick_values[(minor_tick_values >= z_bounds[0]) & (minor_tick_values <= z_bounds[1])], minor=True)

        cbar.ax.tick_params(which="major", labelsize=fontsize - 3, direction="in",
                            color=plot_settings["colorbar_major_ticks_color"],
                            width=plot_settings["colorbar_major_ticks_width"],
                            length=plot_settings["colorbar_major_ticks_length"],
                            pad=plot_settings["colorbar_major_ticks_pad"])
        cbar.ax.tick_params(which="minor", labelsize=fontsize - 3, direction="in",
                            color=plot_settings["colorbar_minor_ticks_color"],
                            width=plot_settings["colorbar_minor_ticks_width"],
                            length=plot_settings["colorbar_minor_ticks_length"],
                            pad=plot_settings["colorbar_minor_ticks_pad"])

        if len(labels) > 2:
            cbar_label_text = labels[2]
        else:
            cbar_label_text = "Profile likelihood" # Default

        if z_is_loglike and not plot_likelihood_ratio:
            cbar_label_text = r"$\ln L - \ln L_{\mathrm{max}}$"
        elif z_is_loglike and plot_likelihood_ratio:
            cbar_label_text = r"Profile likelihood ratio $\Lambda = L/L_{\mathrm{max}}$"

        cbar.set_label(cbar_label_text, fontsize=plot_settings["colorbar_label_fontsize"],
                       labelpad=plot_settings["colorbar_label_pad"],
                       rotation=plot_settings["colorbar_label_rotation"])

    # Return plot objects
    return fig, ax, cbar_ax



def plot_conditional_credible_intervals(
    x_data: np.ndarray, y_data: np.ndarray, posterior_weights: np.ndarray,
    labels: tuple, n_bins: tuple, xy_bounds=None,
    credible_regions=[],
    draw_interval_connectors=True,
    draw_interval_limits=True,
    add_mean_posterior_marker=True,
    add_max_posterior_marker=True,
    shaded_credible_region_bands=False,
    x_condition="bin",
    missing_value_color=None,
    plot_settings=gambit_plot_settings.plot_settings):

    # Make local copies
    xy_bounds = deepcopy(xy_bounds) if xy_bounds is not None else None

    cbar_ax_to_return = None

    known_x_conditions = ["bin", "upperbound", "lowerbound"]
    if x_condition not in known_x_conditions:
        raise Exception(f"Argument 'x_condition' must be one of {', '.join(known_x_conditions)}.")

    # Determine bounds
    if xy_bounds is None:
        # Determine bounds from data if not provided
        x_min_data, x_max_data = np.min(x_data), np.max(x_data)
        y_min_data, y_max_data = np.min(y_data), np.max(y_data)
        xy_bounds = ([x_min_data, x_max_data], [y_min_data, y_max_data])

    xy_bounds[0][0] -= np.finfo(float).eps
    xy_bounds[0][1] += np.finfo(float).eps
    xy_bounds[1][0] -= np.finfo(float).eps
    xy_bounds[1][1] += np.finfo(float).eps
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    # Define x-bins
    x_bin_limits = np.linspace(x_min, x_max, n_bins[0] + 1)

    # Initialize Z_values_2D for posterior probabilities
    if plot_settings["interpolation"]:
        Z_values_2D = np.full((plot_settings["interpolation_resolution"], n_bins[0]), np.nan)
        # y_coarse is implicitly defined by plot_details["main_graph_x_data"] later
        y_fine = np.linspace(y_min, y_max, plot_settings["interpolation_resolution"])
    else:
        Z_values_2D = np.full((n_bins[1], n_bins[0]), np.nan)


    cr_boundaries = []
    mean_posterior_coordinates = []
    max_posterior_coordinates = []

    for i in range(n_bins[0]):
        # Select data for current x-bin
        current_x_min = x_bin_limits[i]
        current_x_max = x_bin_limits[i+1]

        if x_condition == "bin":
            if i == n_bins[0] - 1: # Handle the right edge of the last bin
                mask = (x_data >= current_x_min) & (x_data <= current_x_max)
            else:
                mask = (x_data >= current_x_min) & (x_data < current_x_max)
        elif x_condition == "upperbound":
            mask = (x_data <= current_x_max)
        elif x_condition == "lowerbound":
            mask = (x_data >= current_x_min)

        y_subset = deepcopy(y_data[mask])
        posterior_weights_subset = deepcopy(posterior_weights[mask])

        # If no data, continue to next x bin
        if len(y_subset) == 0 or np.sum(posterior_weights_subset) == 0:
            if add_mean_posterior_marker:
                 mean_posterior_coordinates.append(np.nan)
            cr_boundaries.append([])
            continue

        # Call plot_1D_posterior to get details for the current slice
        _, _, plot_details = plot_1D_posterior(
            x_data=y_subset,
            posterior_weights=posterior_weights_subset,
            x_label=labels[1], # y-axis of the conditional plot is x-axis of the 1D posterior
            n_bins=n_bins[1],
            x_bounds=xy_bounds[1], # y-bounds of conditional plot are x-bounds for 1D posterior
            credible_regions=credible_regions,
            plot_relative_probability=True,
            add_mean_posterior_marker=add_mean_posterior_marker,
            add_max_posterior_marker=add_max_posterior_marker,
            fill_color_below_graph=False,
            shaded_credible_region_bands=True,
            plot_settings=plot_settings,
            return_plot_details=True
        )
        plt.close()  # plot_1D_posterior creates new figures, so we should close them

        # Get the Z data. The x coordinates from the 1D plot are the 
        # bin centers for the y-axis of our plot.
        use_z = plot_details["main_graph_y_data"]
        if plot_settings["interpolation"]:
            # use_z contains the posterior value at the *bin centres* of the 1D posterior, 
            # while "main_graph_x_data" has the bin edges. So we must compute the bin centres.
            y_coarse = 0.5 * (plot_details["main_graph_x_data"][:-1] + plot_details["main_graph_x_data"][1:])
            # Now interpolate
            use_z = np.interp(y_fine, y_coarse, use_z, left=0, right=0)
        Z_values_2D[:, i] = use_z

        # Get credible interval boundaries
        cr_boundaries.append(plot_details["cl_fill_between_coordinates"])

        # Get marker position
        if add_mean_posterior_marker:
            if "mean_posterior_coordinate" in plot_details:
                mean_posterior_coordinates.append(plot_details["mean_posterior_coordinate"])
            else:
                mean_posterior_coordinates.append(np.nan)
        if add_max_posterior_marker:
            if "max_posterior_coordinate" in plot_details:
                max_posterior_coordinates.append(plot_details["max_posterior_coordinate"])
            else:
                max_posterior_coordinates.append(np.nan)

    # Create an empty figure using our plot settings
    if shaded_credible_region_bands:
        fig, ax = create_empty_figure_1D(xy_bounds, plot_settings)
    else:
        if missing_value_color is None:
            missing_value_color = plot_settings["facecolor_plot"]
        fig, ax = create_empty_figure_2D(xy_bounds, plot_settings, use_facecolor=missing_value_color)


    # Plot 2D color plot?
    if not shaded_credible_region_bands:

        # Create color normalization for posterior
        posterior_prob_bounds = (0.0, 1.0) # Relative probability
        norm = matplotlib.cm.colors.Normalize(vmin=posterior_prob_bounds[0], vmax=posterior_prob_bounds[1])

        im = ax.imshow(
            Z_values_2D,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            aspect="auto",
            cmap=plot_settings["colormap"],
            interpolation="none", # or "nearest" if preferred for binned data
            norm=norm,
        )

    # Loop over x bins to draw intervals, markers, etc.
    for i in range(n_bins[0]):
        current_x_min = x_bin_limits[i]
        current_x_max = x_bin_limits[i+1]
        current_x_bin_centre = 0.5 * (current_x_min + current_x_max)

        if i >= len(cr_boundaries): # Skip if no data for this bin
            continue
        x_bin_cr_bounds = cr_boundaries[i]

        for cr_idx, cr_dict in enumerate(x_bin_cr_bounds):
            starts_y_coords = cr_dict["fill_starts_x"]
            ends_y_coords = cr_dict["fill_ends_x"]

            if draw_interval_limits:
                use_color = plot_settings["contour_colors"][cr_idx % len(plot_settings["contour_colors"])]
                use_linewidth = plot_settings["contour_linewidths"][cr_idx % len(plot_settings["contour_linewidths"])]
                use_linestyle = plot_settings["contour_linestyles"][cr_idx % len(plot_settings["contour_linestyles"])]

            assert(len(starts_y_coords) == len(ends_y_coords))
            for k in range(len(starts_y_coords)):
                start_y = starts_y_coords[k]
                end_y = ends_y_coords[k]

                if shaded_credible_region_bands:
                    plt.fill_between(
                        x=[current_x_min, current_x_max],
                        y1=[start_y, start_y],
                        y2=[end_y, end_y],
                        color=plot_settings["1D_posterior_color"],
                        alpha=plot_settings["1D_posterior_fill_alpha"],
                        linewidth=0.0,
                        zorder=0,
                    )

                if draw_interval_limits:
                    plt.plot(
                        [current_x_min, current_x_max], [start_y, start_y],
                        color=use_color,
                        linewidth=use_linewidth,
                        linestyle=use_linestyle,
                        zorder=0,
                    )
                    plt.plot(
                        [current_x_min, current_x_max], [end_y, end_y],
                        color=use_color,
                        linewidth=use_linewidth,
                        linestyle=use_linestyle,
                        zorder=0,
                )
                if draw_interval_connectors:
                    plt.plot(
                        [current_x_bin_centre, current_x_bin_centre], [start_y, end_y],
                        color=plot_settings["connector_color"],
                        linewidth=plot_settings["connector_linewidth"],
                        linestyle=plot_settings["connector_linestyle"],
                        zorder=0,
                    )

        if add_mean_posterior_marker and not np.isnan(mean_posterior_coordinates[i]):
            mean_post_y_coord = mean_posterior_coordinates[i]
            ax.scatter(current_x_bin_centre, mean_post_y_coord,
                       marker=plot_settings["posterior_mean_marker"],
                       s=plot_settings["posterior_mean_marker_size"],
                       c=plot_settings["posterior_mean_marker_color"],
                       edgecolor=plot_settings["posterior_mean_marker_edgecolor"],
                       linewidth=plot_settings["posterior_mean_marker_linewidth"],
                       zorder=100, clip_on=False)

        if add_max_posterior_marker and not np.isnan(max_posterior_coordinates[i]):
            max_post_y_coord = max_posterior_coordinates[i]
            ax.scatter(current_x_bin_centre, max_post_y_coord,
                       marker=plot_settings["posterior_max_marker"],
                       s=plot_settings["posterior_max_marker_size"],
                       c=plot_settings["posterior_max_marker_color"],
                       edgecolor=plot_settings["posterior_max_marker_edgecolor"],
                       linewidth=plot_settings["posterior_max_marker_linewidth"],
                       zorder=100, clip_on=False)

        # Add vertical separators
        if i > 0:
            plt.plot(
                [current_x_min, current_x_min], [y_min, y_max],
                color=plot_settings["separator_color"],
                linewidth=plot_settings["separator_linewidth"],
                linestyle="solid",
                zorder=1,
            )

    # Set axis labels
    x_label = labels[0]
    if x_condition == "upperbound":
        x_label = f"{x_label}, upper bound"
    elif x_condition == "lowerbound":
        x_label = f"{x_label}, lower bound"
    y_label = labels[1]
    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Add colorbar
    if not shaded_credible_region_bands:
        cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"],
                             loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])
        cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
        cbar.outline.set_linewidth(plot_settings["framewidth"])

        # Use posterior_prob_bounds for ticks
        cbar.set_ticks(np.linspace(posterior_prob_bounds[0], posterior_prob_bounds[1], plot_settings["colorbar_n_major_ticks"]), minor=False)
        minor_tick_values = np.linspace(posterior_prob_bounds[0], posterior_prob_bounds[1], plot_settings["colorbar_n_minor_ticks"])
        cbar.set_ticks(minor_tick_values[(minor_tick_values >= posterior_prob_bounds[0]) & (minor_tick_values <= posterior_prob_bounds[1])], minor=True)

        cbar.ax.tick_params(which="major", labelsize=fontsize - 3, direction="in",
                            color=plot_settings["colorbar_major_ticks_color"],
                            width=plot_settings["colorbar_major_ticks_width"],
                            length=plot_settings["colorbar_major_ticks_length"],
                            pad=plot_settings["colorbar_major_ticks_pad"])
        cbar.ax.tick_params(which="minor", labelsize=fontsize - 3, direction="in",
                            color=plot_settings["colorbar_minor_ticks_color"],
                            width=plot_settings["colorbar_minor_ticks_width"],
                            length=plot_settings["colorbar_minor_ticks_length"],
                            pad=plot_settings["colorbar_minor_ticks_pad"])

        cbar_label_text = r"Relative probability $P/P_{\mathrm{max}}$"
        cbar.set_label(cbar_label_text, fontsize=plot_settings["colorbar_label_fontsize"],
                       labelpad=plot_settings["colorbar_label_pad"],
                       rotation=plot_settings["colorbar_label_rotation"])
        cbar_ax_to_return = cbar_ax

    # Return plot objects
    return fig, ax, cbar_ax_to_return



def plot_1D_posterior(x_data: np.ndarray, posterior_weights: np.ndarray,
                      x_label: str, n_bins: tuple, x_bounds = None, 
                      credible_regions = [], plot_relative_probability = True, 
                      add_mean_posterior_marker = True,
                      add_max_posterior_marker = True,
                      fill_color_below_graph=False,
                      shaded_credible_region_bands = True,
                      plot_settings = gambit_plot_settings.plot_settings,
                      return_plot_details = False,
                      graph_coordinates_output_file=None) -> None:

    # Make local copies
    x_bounds = deepcopy(x_bounds) if x_bounds is not None else None

    # Initialize plot_details dict
    if return_plot_details:
        plot_details = {}

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
        y_label = r"Relative probability $P/P_{\mathrm{max}}$"

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Make a histogram of the 1D posterior distribution
    y_data = histogram
    if plot_relative_probability:
        y_data = y_data / np.max(y_data)

    # Store histogram data
    if return_plot_details:
        plot_details["main_graph_x_data"] = x_edges
        plot_details["main_graph_y_data"] = y_data

    if fill_color_below_graph:
        plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="stepfilled", color=plot_settings["1D_posterior_color"], alpha=plot_settings["1D_posterior_fill_alpha"])
    hist_vals, bins, _ = plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="step", color=plot_settings["1D_posterior_color"])

    if graph_coordinates_output_file is not None:
        # Construct step coords from hist_vals and bins
        x_coords = np.empty(2 * len(hist_vals))
        x_coords[0::2] = bins[:-1]
        x_coords[1::2] = bins[1:]
        y_coords = np.repeat(hist_vals, 2)
        np.savetxt(graph_coordinates_output_file, np.column_stack((x_coords, y_coords)), delimiter=',', header="#x, y", comments="")
        print(f"Wrote file: {graph_coordinates_output_file}")

    # Draw credible region lines?
    if len(credible_regions) > 0:

        # Initialize cl_lines_y_vals and cl_fill_between_coordinates
        if return_plot_details:
            plot_details["cl_lines_y_vals"] = []
            if shaded_credible_region_bands:
                plot_details["cl_fill_between_coordinates"] = []

        # For each requested CR line, find the posterior 
        # density height at which to draw the line. 
        sorted_hist = np.sort(y_data)[::-1]
        cumulative_sum = np.cumsum(sorted_hist)
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        
        linewidths = cycle(list(plot_settings["contour_linewidths"]))
        
        for cr in credible_regions:
            line_y_val = sorted_hist[np.searchsorted(normalized_cumulative_sum, cr)]
            # Store credible region line y-value
            if return_plot_details:
                plot_details["cl_lines_y_vals"].append(line_y_val)

            ax.plot([x_min, x_max], [line_y_val, line_y_val], color=plot_settings["1D_posterior_color"], linewidth=next(linewidths), linestyle="dashed")
            cr_text = f"${100*cr:.1f}\\%\\,$CR"
            ax.text(0.06, line_y_val, cr_text, ha="left", va="bottom", fontsize=plot_settings["header_fontsize"], 
                    color=plot_settings["1D_posterior_color"], transform = ax.transAxes)

            if shaded_credible_region_bands:
                new_y_data_shaded = deepcopy(y_data) # Use a different variable name to avoid confusion
                new_y_data_shaded[new_y_data_shaded < line_y_val] = 0.0
                plt.hist(x_edges[:-1], n_bins, weights=new_y_data_shaded, histtype="stepfilled", color=plot_settings["1D_posterior_color"], alpha=plot_settings["1D_posterior_fill_alpha"])

                # Store fill coordinates for shaded bands
                if return_plot_details:
                    current_cl_fill_info = {"fill_starts_x": [], "fill_ends_x": [], "fill_starts_y": [], "fill_ends_y": []}
                    in_segment = False
                    for i in range(len(y_data)):
                        if y_data[i] >= line_y_val and not in_segment:
                            # Start of a new segment
                            in_segment = True
                            current_cl_fill_info["fill_starts_x"].append(x_edges[i])
                            current_cl_fill_info["fill_starts_y"].append(line_y_val) # Or y_data[i] if top of segment start
                        elif y_data[i] < line_y_val and in_segment:
                            # End of a segment
                            in_segment = False
                            current_cl_fill_info["fill_ends_x"].append(x_edges[i]) # x_edges[i] is the right edge of bin i-1, which is correct here
                            current_cl_fill_info["fill_ends_y"].append(line_y_val) # Or y_data[i-1] if top of segment end

                    # If the last segment extends to the end of the histogram
                    if in_segment:
                        current_cl_fill_info["fill_ends_x"].append(x_edges[-1])
                        current_cl_fill_info["fill_ends_y"].append(line_y_val) # Or y_data[-1]

                    plot_details["cl_fill_between_coordinates"].append(current_cl_fill_info)

    # Add marker at the mean posterior point
    if add_mean_posterior_marker:
        x_post_mean_marker = np.average(x_data, weights=posterior_weights)
        y_post_mean_marker = 0.0
        ax.scatter(x_post_mean_marker, y_post_mean_marker, marker=plot_settings["posterior_mean_marker"], s=plot_settings["posterior_mean_marker_size"], c=plot_settings["posterior_mean_marker_color"],
                   edgecolor=plot_settings["posterior_mean_marker_edgecolor"], linewidth=plot_settings["posterior_mean_marker_linewidth"], zorder=100, clip_on=False)
        # Store mean posterior marker coordinate
        if return_plot_details:
            plot_details["mean_posterior_coordinate"] = x_post_mean_marker

    # Add marker at the max posterior point
    if add_max_posterior_marker:
        x_post_max_marker = x_centers[np.argmax(histogram)]
        y_post_max_marker = 0.0
        ax.scatter(x_post_max_marker, y_post_max_marker, marker=plot_settings["posterior_max_marker"], s=plot_settings["posterior_max_marker_size"], c=plot_settings["posterior_max_marker_color"],
                   edgecolor=plot_settings["posterior_max_marker_edgecolor"], linewidth=plot_settings["posterior_max_marker_linewidth"], zorder=100, clip_on=False)
        # Store max posterior marker coordinate
        if return_plot_details:
            plot_details["max_posterior_coordinate"] = x_post_max_marker

    # Return plot
    if return_plot_details:
        return fig, ax, plot_details
    else:
        return fig, ax



def plot_2D_posterior(x_data: np.ndarray, y_data: np.ndarray, posterior_weights: np.ndarray, 
                      labels: tuple, n_bins: tuple, xy_bounds = None, 
                      credible_regions = [], 
                      contour_coordinates_output_file = None,
                      plot_relative_probability = True, 
                      add_mean_posterior_marker = True,
                      color_data: np.ndarray = None,
                      color_point_estimate: str = "mean",
                      color_posterior_bins: int = 20,
                      color_label: str = None,
                      color_bounds = None,
                      color_within_credible_region = None,
                      missing_value_color = None,
                      plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Make local copies
    xy_bounds = deepcopy(xy_bounds) if xy_bounds is not None else None

    # Sanity checks
    if not (x_data.shape == y_data.shape == posterior_weights.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == len(posterior_weights.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    if color_data is not None:
        if not (color_data.shape == x_data.shape):
            raise Exception("Input array color_data must have the same shape as x_data, y_data, posterior_weights.")
        if not (len(color_data.shape) == 1):
            raise Exception("Input array color_data must be one-dimensional.")

    if color_point_estimate not in ["mean", "maxpost"]:
        raise Exception("The argument 'color_point_estimate' must be either 'mean' or 'maxpost'.")

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

    # Initialize z_data for imshow
    z_data_for_imshow = histogram.T
    if plot_relative_probability and color_data is None:
        z_data_for_imshow = z_data_for_imshow / np.max(z_data_for_imshow)

    # Colorbar range
    cmap_vmin = 0.0
    cmap_vmax = None
    if color_data is None:
        cmap_vmax = np.max(histogram)
        if plot_relative_probability:
            cmap_vmax = 1.0

    # Logic for calculating color_values_for_bins, if color_data is not None
    if color_data is not None:
        color_values_for_bins = np.full_like(histogram, np.nan, dtype=float)
        num_color_bins_1d = color_posterior_bins
        color_bins_1d = np.linspace(color_bounds[0], color_bounds[1], num_color_bins_1d + 1)

        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                x_bin_min, x_bin_max = x_edges[i], x_edges[i+1]
                y_bin_min, y_bin_max = y_edges[j], y_edges[j+1]

                # Create a boolean mask for samples within the current 2D bin
                mask = (x_data >= x_bin_min) & (x_data < x_bin_max) & \
                       (y_data >= y_bin_min) & (y_data < y_bin_max)

                # No samples in this bin? Continue to next bin
                if not np.any(mask):
                    continue

                color_data_subset = color_data[mask]
                weights_for_color_data_subset = posterior_weights[mask]
                weights_sum = np.sum(weights_for_color_data_subset)

                # All zero weights? Continue to next bin
                if weights_sum == 0:
                    continue

                weights_for_color_data_subset = weights_for_color_data_subset / weights_sum

                if color_point_estimate == "mean":
                    color_values_for_bins[i, j] = np.sum(weights_for_color_data_subset * color_data_subset)

                elif color_point_estimate == "maxpost":
                    color_hist, color_bin_edges = np.histogram(
                        color_data_subset,
                        bins=color_bins_1d,
                        weights=weights_for_color_data_subset,
                        density=False
                    )
                    peak_color_bin_index = np.argmax(color_hist)
                    peak_color_value = (color_bin_edges[peak_color_bin_index] + color_bin_edges[peak_color_bin_index+1]) / 2.0
                    color_values_for_bins[i, j] = peak_color_value


        # Mask some of the colored data?
        if color_within_credible_region is not None:
            # Find the contour level corresponding to the given credible region
            sorted_hist = np.sort(histogram.ravel())[::-1]
            cumulative_sum = np.cumsum(sorted_hist)
            normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
            contour_level = sorted_hist[np.searchsorted(normalized_cumulative_sum, color_within_credible_region)]
            # For posterior bins outside the given CR contour, set color value to nan
            mask = histogram < contour_level
            color_values_for_bins[mask] = np.nan 

        # Set the z data for imshow
        z_data_for_imshow = color_values_for_bins.T

        # Update cmap_vmin and cmap_vmax for color_data
        if color_bounds is not None:
            cmap_vmin = color_bounds[0]
            cmap_vmax = color_bounds[1]
        else:
            cmap_vmin = np.nanmin(color_values_for_bins)
            cmap_vmax = np.nanmax(color_values_for_bins)

    # Create an empty figure using our plot settings
    if missing_value_color is None:
        if color_data is None:
            missing_value_color = plot_settings["colormap"](0.0)
        else:
            missing_value_color = plot_settings["facecolor_plot"]
    fig, ax = create_empty_figure_2D(xy_bounds, plot_settings, use_facecolor=missing_value_color)

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
    # z_data for imshow is z_data_for_imshow, computed above
    im = ax.imshow(z_data_for_imshow, aspect="auto", extent=[x_min, x_max, y_min, y_max],
                   cmap=plot_settings["colormap"], norm=norm, origin="lower")

    # Draw credible region contours?
    contour_levels = []
    if len(credible_regions) > 0:

        # For each requested CR contour, find the posterior 
        # density height at which to draw the contour using the original histogram.
        sorted_hist = np.sort(histogram.ravel())[::-1]
        cumulative_sum = np.cumsum(sorted_hist)
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        for cr in credible_regions:
            contour_levels.append(sorted_hist[np.searchsorted(normalized_cumulative_sum, cr)])

        contour_levels.sort()

        if contour_levels: # Only draw contours if levels were actually computed
            contour = ax.contour(X, Y, histogram.T, contour_levels, colors=plot_settings["contour_colors"],
                                 linewidths=plot_settings["contour_linewidths"], linestyles=plot_settings["contour_linestyles"])

            # Save contour coordinates to file?
            if contour_coordinates_output_file != None:
                header = "# x,y coordinates for contours corresponding to the "  + ", ".join([f"{cr:.4e}" for cr in credible_regions]) + " credible regions. Sets of coordinates for individual closed contours are separated by nan entries."
                save_contour_coordinates(contour, contour_coordinates_output_file, header=header)

    # Add marker at the mean posterior point?
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
    # Ensure minor ticks are within the bounds
    minor_tick_values = np.linspace(cmap_vmin, cmap_vmax, plot_settings["colorbar_n_minor_ticks"])
    cbar.set_ticks(minor_tick_values[(minor_tick_values >= cmap_vmin) & (minor_tick_values <= cmap_vmax)], minor=True)

    cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
    cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

    # Colorbar label for color_data
    cbar_label_str = ""
    if color_data is not None:
        if color_label is not None:
            cbar_label_str = color_label
        else:
            cbar_label_str = "[Missing label]"

        if color_point_estimate == "mean":
            cbar_label_str = f"{cbar_label_str}, posterior mean"
        elif color_point_estimate == "maxpost":
            cbar_label_str = f"{cbar_label_str}, posterior max"

    else:
        cbar_label_str = "Posterior probability"
        if plot_relative_probability:
            cbar_label_str = r"Relative probability $P/P_{\mathrm{max}}$"

    cbar.set_label(cbar_label_str, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax


def plot_2D_scatter(x_data: np.ndarray, y_data: np.ndarray, labels: tuple, xy_bounds = None,
                    sort_data: np.ndarray = None, reverse_sort: bool = False,
                    color_data: np.ndarray = None, color_label: str = None,
                    color_bounds = None, 
                    plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Make local copies
    x_data = np.copy(x_data)
    y_data = np.copy(y_data)
    if sort_data is not None:
        sort_data = np.copy(sort_data)
    if color_data is not None:
        color_data = np.copy(color_data)
    xy_bounds = deepcopy(xy_bounds) if xy_bounds is not None else None

    # Sanity checks
    if not (x_data.shape == y_data.shape):
        raise Exception("Input arrays x_data, y_data must have the same shape.")
    if sort_data is not None and not (x_data.shape == sort_data.shape):
        raise Exception("Input array sort_data must have the same shape as x_data and y_data.")
    if color_data is not None and not (x_data.shape == color_data.shape):
        raise Exception("Input array color_data must have the same shape as x_data and y_data.")

    if not (len(x_data.shape) == 1 and len(y_data.shape) == 1):
        raise Exception("Input arrays x_data, y_data must be one-dimensional.")
    if sort_data is not None and not (len(sort_data.shape) == 1):
        raise Exception("Input array z_data must be one-dimensional.")
    if color_data is not None and not (len(color_data.shape) == 1):
        raise Exception("Input array color_data must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Sorting data
    # If sort_data is provided, sort all data based on sort_data.
    # If color_data is also provided, it's sorted along with x and y.
    # If sort_data is not provided, no sorting is done.
    if sort_data is not None:
        if reverse_sort:
            p = np.argsort(sort_data)[::-1] # Descending
        else:
            p = np.argsort(sort_data)# Ascending
        x_data = x_data[p]
        y_data = y_data[p]
        sort_data = sort_data[p]
        if color_data is not None:
            color_data = color_data[p]

    # Plot bounds in x and y
    if xy_bounds is None:
        xy_bounds = ([np.min(x_data), np.max(x_data)], [np.min(y_data), np.max(y_data)])
    # Add epsilon padding to avoid points landing exactly on the boundary
    xy_bounds[0][0] -= np.finfo(float).eps
    xy_bounds[0][1] += np.finfo(float).eps
    xy_bounds[1][0] -= np.finfo(float).eps
    xy_bounds[1][1] += np.finfo(float).eps
    x_min, x_max = xy_bounds[0]
    y_min, y_max = xy_bounds[1]

    # Color data bounds and label
    if color_bounds is None:
        color_bounds = (np.min(color_data), np.max(color_data))

    if color_label is None:
        color_label = "[Missing label]"

    # Create empty figure
    fig, ax = create_empty_figure_2D(xy_bounds, plot_settings)

    # Axis labels
    x_label = labels[0]
    y_label = labels[1]

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Create a color scale normalization if color data is present
    norm = None
    if color_data is not None:
        norm = matplotlib.cm.colors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])

    # Make the scatter plot
    scatter_plot_args = {
        "s": plot_settings["scatter_marker_size"],
        "cmap": plot_settings["colormap"],
        "norm": norm,
        "rasterized": True,
        "marker": plot_settings["scatter_marker"],
        "edgecolors": plot_settings["scatter_marker_edgecolor"],
        "linewidth": plot_settings["scatter_marker_edgewidth"],
    }
    if color_data is not None:
         scatter_plot_args["c"] = color_data
    else:
        scatter_plot_args["color"] = plot_settings["scatter_marker_color"]
        del scatter_plot_args["cmap"]
        del scatter_plot_args["norm"]

    im = ax.scatter(x_data, y_data, **scatter_plot_args)

    # Add a colorbar if color data was used
    cbar_ax = None
    if color_data is not None:
        cbar_ax = inset_axes(ax, width=plot_settings["colorbar_width"], height=plot_settings["colorbar_height"],
                             loc=plot_settings["colorbar_loc"], borderpad=plot_settings["colorbar_borderpad"])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation=plot_settings["colorbar_orientation"])

        cbar.outline.set_edgecolor(plot_settings["framecolor_colorbar"])
        cbar.outline.set_linewidth(plot_settings["framewidth"])

        if color_bounds is not None:
            cbar.set_ticks(np.linspace(color_bounds[0], color_bounds[1], plot_settings["colorbar_n_major_ticks"]), minor=False)
            minor_tick_values = np.linspace(color_bounds[0], color_bounds[1], plot_settings["colorbar_n_minor_ticks"])
            cbar.set_ticks(minor_tick_values[(minor_tick_values >= color_bounds[0]) & (minor_tick_values <= color_bounds[1])], minor=True)

        cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
        cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

        cbar.set_label(color_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax


def nearest_neighbor_averaging(hdf5_file_and_group_names, target_dataset, NN_instance, 
    parameter_dataset_tag="::primary_parameters::", scaler=None, filter_invalid_points=True):

    # Create a list of tuples of the form (shorthand key, (full dataset name, dataset type)).
    # First add the target dataset to be averaged
    datasets = [
        (target_dataset, (target_dataset, float)),
    ]
    # Then add the datasets for all the model parameters, assuming all files contain the same datasets
    all_dataset_names = collect_all_dataset_names(hdf5_file_and_group_names[0])
    i = 0
    shorthand_param_names = []
    for dset_name in all_dataset_names:
        if parameter_dataset_tag in dset_name:
            datasets.append( (f"x{i}", (dset_name, float)) )
            shorthand_param_names.append(f"x{i}")
            i += 1
    n_pars = i

    # Now create our main data dictionary by reading the hdf5 files
    data = read_hdf5_datasets(hdf5_file_and_group_names, datasets, filter_invalid_points=filter_invalid_points)
    n_points = len(data[target_dataset])

    # Construct X array from input parameter datasets

    # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

    X = np.array([data[par_name] for par_name in shorthand_param_names]).T

    # Scale the input coordinates before doing the nearest-neighbors search?
    if scaler != None:
        X = scaler.fit_transform(X)

    # Now do the nearest-neighbors search
    neighbors = NN_instance.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Compute the mean target_dataset value for each group and return
    mean_target_per_group = np.zeros(n_points)
    for i in range(n_points):
        group_indices = indices[i]
        mean_target_per_group[i] = np.mean(data[target_dataset][group_indices])

    return mean_target_per_group



def grid_2D_interpolation(x_values, y_values, target_values, interpolation_resolution):

    from scipy.interpolate import RegularGridInterpolator

    # Created an array with all nan entries replaced using averages of nearby non-nan entries.
    # This "extended" array will be used to avoid surprising interpolation results for the elements 
    # in the target_values array that have neighboring nan entries. 
    target_values_extended = target_values.copy()
    while np.isnan(target_values_extended).sum() > 0:
        target_values_extended = fill_nan_with_neighbor_mean(target_values_extended)

    # Create two interpolators: One using linear interpolation with the extended data, 
    # and another using nearest-neighbor interpolation with the original data.
    interpolator_extended = RegularGridInterpolator((x_values[0,:], y_values[:,0]), target_values_extended.T, method="linear", fill_value=np.nan)
    interpolator = RegularGridInterpolator((x_values[0,:], y_values[:,0]), target_values.T, method="nearest", fill_value=np.nan)

    # Evaluate the two interpolators on the new grid of interpolation points
    xi = np.linspace(np.min(x_values[0,:]), np.max(x_values[0,:]), interpolation_resolution)
    yi = np.linspace(np.min(y_values[:,0]), np.max(y_values[:,0]), interpolation_resolution)
    Xi, Yi = np.meshgrid(xi, yi, indexing='ij')
    xi_yi_points = np.stack([Xi.ravel(), Yi.ravel()], axis=-1)

    target_values_interpolated = interpolator(xi_yi_points).reshape(Xi.shape)
    target_values_interpolated_extended = interpolator_extended(xi_yi_points).reshape(Xi.shape)

    # Finally, use target_values_interpolated to mask those parts of 
    # target_values_interpolated_extended that where nan entries originally.
    target_values_interpolated_extended[np.isnan(target_values_interpolated)] = np.nan

    return Xi, Yi, target_values_interpolated_extended


def fill_nan_with_neighbor_mean(arr):
    orig = np.array(arr, dtype=float)
    result = orig.copy()
    nrows, ncols = orig.shape

    # All 8 neighbor offsets
    neighbor_offsets = [
        (-1, -1), (-1,  0), (-1, +1),
        ( 0, -1),           ( 0, +1),
        (+1, -1), (+1,  0), (+1, +1),
    ]

    # Do one sweep over the array
    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(orig[i, j]):
                neighbor_vals = []
                for di, dj in neighbor_offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nrows and 0 <= nj < ncols:
                        val = orig[ni, nj]
                        if not np.isnan(val):
                            neighbor_vals.append(val)
                if neighbor_vals:
                    result[i, j] = np.mean(neighbor_vals)

    return result



