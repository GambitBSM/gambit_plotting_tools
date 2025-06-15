from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_header


#
# Read file 
#

hdf5_file = "./example_data/results_multinest.hdf5"
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("mu",      ("#NormalDist_parameters @NormalDist::primary_parameters::mu", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
    ("Posterior", ("Posterior", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)


# 
# Make a plot of conditional credible intervals
# 

# Get credible regions
credible_regions = [0.954]

# Plot variables
x_key = "mu"
y_key = "sigma"
posterior_weights_key = "Posterior"

# Set some bounds manually?
dataset_bounds = {
    "mu": [19.5, 22.0],
    "sigma": [1.0, 4.0],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": r"$\mu$ (unit)",
    "sigma": r"$\sigma$ (unit)",
}

# Number of bins used for profiling/histogramming
xy_bins = (20, 20)

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

plot_settings["interpolation"] = True
plot_settings["interpolation_resolution"] = 200

plot_settings["separator_linewidth"] = 3 * plot_settings["framewidth"]
plot_settings["separator_color"] = "black"

plot_settings["posterior_mean_marker"] = "o"
plot_settings["posterior_mean_marker_size"] = 10
plot_settings["posterior_mean_marker_linewidth"] = 0.8

plot_settings["posterior_max_marker"] = "D"
plot_settings["posterior_max_marker_size"] = 10
plot_settings["posterior_max_marker_linewidth"] = 0.8

plot_settings["contour_linewidth"] = 1.0
plot_settings["contour_color"] = "white"
plot_settings["contour_linestyle"] = "solid"

plot_settings["connector_linewidth"] = plot_settings["contour_linewidth"] * 0.5
plot_settings["connector_color"] = plot_settings["contour_color"]
plot_settings["connector_linestyle"] = "dotted"

plot_settings["colormap"] = matplotlib.colormaps["inferno"]

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

fig, ax, cbar_ax = plot_utils.plot_conditional_credible_intervals(
    data[x_key],
    data[y_key],
    data[posterior_weights_key],
    labels,
    xy_bins,
    xy_bounds=xy_bounds,
    credible_regions=credible_regions,
    draw_interval_connectors=True,
    add_mean_posterior_marker=True,
    add_max_posterior_marker=False,
    shaded_credible_region_bands=False,
    missing_value_color=plot_settings["colormap"](0.0),
    x_condition="upperbound",
    plot_settings=plot_settings,
)

# Header text
header_text = r"Conditional $2\sigma$ credible intervals."
if plt.rcParams.get("text.usetex"):
    header_text += r" \textsf{GAMBIT} 2.5"
else:
    header_text += r" GAMBIT 2.5"
add_header(header_text, ax=ax)

# Save to file
output_path = f"./plots/conditional_credible_intervals__{x_key}__{y_key}__colored.pdf"
plot_utils.create_folders_if_not_exist(output_path)

plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")
