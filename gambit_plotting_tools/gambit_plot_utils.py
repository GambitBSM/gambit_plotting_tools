from copy import deepcopy
from collections import OrderedDict
import os
import shutil
import numpy as np
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

    # Iterate over each contour line
    for path in contour.get_paths():
        vertices = path.vertices
        coordinates.append(vertices)
        # Use NaN separator to mark contour breaks
        coordinates.append(np.array([[np.nan, np.nan]]))

    # for collection in contour.collections:
    #     for path in collection.get_paths():
    #         vertices = path.vertices
    #         coordinates.append(vertices)
    #         # Use NaN separator to mark contour breaks
    #         coordinates.append(np.array([[np.nan, np.nan]]))

    # Concatenate all coordinate arrays into one
    coordinates = np.vstack(coordinates)

    # Save to file
    create_folders_if_not_exist(contour_coordinates_output_file)
    np.savetxt(contour_coordinates_output_file, coordinates, delimiter=',', header=header, comments="")
    print(f"Wrote file: {contour_coordinates_output_file}")



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



def create_empty_figure_2D(xy_bounds, plot_settings):

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



def bin_and_profile_1D(x_data, y_data, n_bins, x_bounds, 
                       already_sorted=False, 
                       y_fill_value=-1*np.finfo(float).max):

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

    # Determine the z value in each bin.
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




def plot_1D_profile(x_data: np.ndarray, y_data: np.ndarray, 
                    x_label: str, n_bins: tuple, x_bounds = None, 
                    confidence_levels = [], y_fill_value = -1*np.finfo(float).max, 
                    y_is_loglike = True, plot_likelihood_ratio = True,
                    add_max_likelihood_marker = True, fill_color_below_graph = True, 
                    shaded_confidence_interval_bands=True,
                    plot_settings = gambit_plot_settings.plot_settings) -> None:

    # Sanity checks
    if not (x_data.shape == y_data.shape):
        raise Exception("All input arrays must have the same shape.")

    if not (len(x_data.shape) == len(y_data.shape) == 1):
        raise Exception("Input arrays must be one-dimensional.")

    # Number of points
    n_pts = x_data.shape[0]

    # Sort data according to y value, from highest to lowest
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


    # Add shaded confidence interval bands?
    if shaded_confidence_interval_bands:

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

            for i in range(len(fill_starts_x)):
                
                x_start, x_end = fill_starts_x[i], fill_ends_x[i]
                y_start, y_end = fill_starts_y[i], fill_ends_y[i]

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

        for i,cl in enumerate(confidence_levels):
            cl_line_y_val = cl_lines_y_vals[i]
            ax.plot([x_min, x_max], [cl_line_y_val, cl_line_y_val], color=plot_settings["1D_profile_likelihood_color"], linewidth=plot_settings["contour_linewidth"], linestyle="dashed")
            cl_text = f"${100*cl:.1f}\\%\\,$CL"
            ax.text(0.06, cl_line_y_val, cl_text, ha="left", va="bottom", fontsize=plot_settings["header_fontsize"], 
                    color=plot_settings["1D_profile_likelihood_color"], transform = ax.transAxes)

    # Add a star at the max-likelihood point
    if (y_is_loglike and add_max_likelihood_marker):
        max_like_index = np.argmax(y_data)
        x_max_like = x_data[max_like_index]
        ax.scatter(x_max_like, 0.0, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                   edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100, clip_on=False)

    # Return plot
    return fig, ax




def plot_2D_profile(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray, 
                    labels: tuple, n_bins: tuple, xy_bounds = None, z_bounds = None,
                    contour_levels = [], contour_coordinates_output_file = None,
                    z_fill_value = -1*np.finfo(float).max, 
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
    x_values, y_values, z_values = bin_and_profile_2D(x_data, y_data, z_data, 
                                                      n_bins, xy_bounds,
                                                      already_sorted=True, 
                                                      z_fill_value=z_fill_value)

    # Convert from lnL - lnL_max = ln(L/Lmax) to L/Lmax 
    if z_is_loglike and plot_likelihood_ratio:
        z_values = np.exp(z_values)

    # Colorbar range
    if z_bounds is None:
        z_bounds = (z_min, z_max)
        if (z_is_loglike) and (plot_likelihood_ratio):
            z_bounds = (0.0, 1.0)
        if (z_is_loglike) and (not plot_likelihood_ratio):
            z_bounds = (z_max - 9.0, z_max)

    # Create an empty figure using our plot settings
    fig, ax = create_empty_figure_2D(xy_bounds, plot_settings)

    # Axis labels
    x_label = labels[0]
    y_label = labels[1]

    fontsize = plot_settings["fontsize"]
    plt.xlabel(x_label, fontsize=fontsize, labelpad=plot_settings["xlabel_pad"])
    plt.ylabel(y_label, fontsize=fontsize, labelpad=plot_settings["ylabel_pad"])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Create a color scale normalization
    norm = matplotlib.cm.colors.Normalize(vmin=z_bounds[0], vmax=z_bounds[1])

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
        contour = ax.contour(x_values, y_values, z_values, contour_levels, colors=plot_settings["contour_color"], 
                             linewidths=[plot_settings["contour_linewidth"]]*len(contour_levels), linestyles=plot_settings["contour_linestyle"])

        # Save contour coordinates to file?
        if contour_coordinates_output_file != None:

            header = "# x,y coordinates for profile likelihood contours at the likelihood ratio values " + ", ".join([f"{l:.4e}" for l in contour_levels]) + ". Sets of coordinates for individual closed contours are separated by nan entries."
            save_contour_coordinates(contour, contour_coordinates_output_file, header=header)

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

    cbar.set_ticks(np.linspace(z_bounds[0], z_bounds[1], plot_settings["colorbar_n_major_ticks"]), minor=False)
    cbar.set_ticks(np.linspace(z_bounds[0], z_bounds[1], plot_settings["colorbar_n_minor_ticks"]), minor=True)

    cbar.ax.tick_params(which="major", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_major_ticks_color"], width=plot_settings["colorbar_major_ticks_width"], length=plot_settings["colorbar_major_ticks_length"], pad=plot_settings["colorbar_major_ticks_pad"])
    cbar.ax.tick_params(which="minor", labelsize=fontsize-3, direction="in", color=plot_settings["colorbar_minor_ticks_color"], width=plot_settings["colorbar_minor_ticks_width"], length=plot_settings["colorbar_minor_ticks_length"], pad=plot_settings["colorbar_minor_ticks_pad"])

    cbar_label = labels[2]
    if (z_is_loglike) and (not plot_likelihood_ratio):
        cbar_label = r"$\ln L   - \ln L_{\mathrm{max}}$"
    if (z_is_loglike) and (plot_likelihood_ratio):
        cbar_label = r"Profile likelihood ratio $\Lambda = L/L_{\mathrm{max}}$"
    cbar.set_label(cbar_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

    # Return plot
    return fig, ax, cbar_ax




def plot_1D_posterior(x_data: np.ndarray, posterior_weights: np.ndarray, 
                      x_label: str, n_bins: tuple, x_bounds = None, 
                      credible_regions = [], plot_relative_probability = True, 
                      add_mean_posterior_marker = True,
                      fill_color_below_graph=False,
                      shaded_credible_region_bands = True,
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

    if fill_color_below_graph:
        plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="stepfilled", color=plot_settings["1D_posterior_color"], alpha=plot_settings["1D_posterior_fill_alpha"])
    plt.hist(x_edges[:-1], n_bins, weights=y_data, histtype="step", color=plot_settings["1D_posterior_color"])

    # Draw credible region lines?
    if len(credible_regions) > 0:

        # For each requested CR line, find the posterior 
        # density height at which to draw the line. 
        sorted_hist = np.sort(y_data)[::-1]
        cumulative_sum = np.cumsum(sorted_hist)
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        for cr in credible_regions:
            line_y_val = sorted_hist[np.searchsorted(normalized_cumulative_sum, cr)]
            ax.plot([x_min, x_max], [line_y_val, line_y_val], color=plot_settings["1D_posterior_color"], linewidth=plot_settings["contour_linewidth"], linestyle="dashed")
            cr_text = f"${100*cr:.1f}\\%\\,$CR"
            ax.text(0.06, line_y_val, cr_text, ha="left", va="bottom", fontsize=plot_settings["header_fontsize"], 
                    color=plot_settings["1D_posterior_color"], transform = ax.transAxes)

            if shaded_credible_region_bands:
                new_y_data = deepcopy(y_data)
                new_y_data[new_y_data < line_y_val] = 0.0
                plt.hist(x_edges[:-1], n_bins, weights=new_y_data, histtype="stepfilled", color=plot_settings["1D_posterior_color"], alpha=plot_settings["1D_posterior_fill_alpha"])

    # Add marker at the mean posterior point
    if add_mean_posterior_marker:
        x_mean = np.average(x_data, weights=posterior_weights)
        y_mean = 0.0
        ax.scatter(x_mean, y_mean, marker=plot_settings["posterior_mean_marker"], s=plot_settings["posterior_mean_marker_size"], c=plot_settings["posterior_mean_marker_color"],
                   edgecolor=plot_settings["posterior_mean_marker_edgecolor"], linewidth=plot_settings["posterior_mean_marker_linewidth"], zorder=100, clip_on=False)

    # Return plot
    return fig, ax



def plot_2D_posterior(x_data: np.ndarray, y_data: np.ndarray, posterior_weights: np.ndarray, 
                      labels: tuple, n_bins: tuple, xy_bounds = None, 
                      credible_regions = [], 
                      contour_coordinates_output_file = None,
                      plot_relative_probability = True, 
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
    fig, ax = create_empty_figure_2D(xy_bounds, plot_settings)

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
        contour = ax.contour(X, Y, histogram.T, contour_levels, colors=plot_settings["contour_color"], 
                             linewidths=[plot_settings["contour_linewidth"]]*len(contour_levels), linestyles=plot_settings["contour_linestyle"])

        # Save contour coordinates to file?
        if contour_coordinates_output_file != None:

            header = "# x,y coordinates for contours corresponding to the "  + ", ".join([f"{cr:.4e}" for cr in credible_regions]) + " credible regions. Sets of coordinates for individual closed contours are separated by nan entries."
            save_contour_coordinates(contour, contour_coordinates_output_file, header=header)

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
        cbar_label = r"Relative probability $P/P_{\mathrm{max}}$"
    cbar.set_label(cbar_label, fontsize=plot_settings["colorbar_label_fontsize"], labelpad=plot_settings["colorbar_label_pad"], rotation=plot_settings["colorbar_label_rotation"])

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







