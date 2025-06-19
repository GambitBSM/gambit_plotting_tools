from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_header


# 
# Read file
# 

hdf5_file = "./example_data/results_run1.hdf5" 
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("mu",      ("#NormalDist_parameters @NormalDist::primary_parameters::mu", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)

# 
# Make a 2D profile likelihood plot
# 

# Plot variables
x_key = "mu"
y_key = "sigma"
z_key = "LogLike"
c_key = "LogLike"

# Set some bounds manually?
dataset_bounds = {
    # "mu": [15, 30],
    # "sigma": [0, 5],
    "mu": [16, 26],
    "sigma": [1, 4],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": r"$\mu$ (unit)",
    "sigma": r"$\sigma$ (unit)",
    "LogLike": r"$\ln L$",
    "color_data": r"Color data",
}

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

plot_settings["facecolor_plot"] = "0.5"

plot_settings["scatter_marker"] = "o"
plot_settings["scatter_marker_size"] = 6
plot_settings["scatter_marker_edgecolor"] = "black"
plot_settings["scatter_marker_edgewidth"] = 0.03

# Discretize colormap?
n_colors = 20
plot_settings["colormap"] = ListedColormap(plot_settings["colormap"](np.linspace(0, 1, n_colors)))

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)
y_label = plot_labels.get(y_key, y_key)
labels = (x_label, y_label)


# Using z_data for sorting and implicitly for coloring as c_data is None
fig, ax, cbar_ax = plot_utils.plot_2D_scatter(
    x_data=data[x_key],
    y_data=data[y_key],
    labels=labels,
    xy_bounds=xy_bounds,
    sort_data=data[z_key],
    reverse_sort=False,
    color_data=data[c_key],
    color_label=plot_labels[c_key],
    color_bounds=[np.max(data[c_key])-20, np.max(data[c_key])],
    plot_settings=plot_settings
)


# Add header showing the number of points
mask = (
    (data[x_key] >= x_bounds[0]) & 
    (data[x_key] <= x_bounds[1]) & 
    (data[y_key] >= y_bounds[0]) & 
    (data[y_key] <= y_bounds[1])
)
n_pts_total = data[x_key].shape[0]
n_pts_in_plot = data[x_key][mask].shape[0]
header_text = f"Scatter plot. Showing {n_pts_in_plot} of {n_pts_total} points." 
if plt.rcParams.get("text.usetex"):
    header_text += r" \textsf{GAMBIT} 2.5"
else:
    header_text += r" GAMBIT 2.5"
add_header(header_text, ax=ax)


# Save to file
output_path = f"./plots/2D_scatter__{x_key}__{y_key}__sortby_{z_key}__color_{c_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Wrote file: {output_path}")
