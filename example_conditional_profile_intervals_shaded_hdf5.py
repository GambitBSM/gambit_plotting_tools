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

# Get contour levels
confidence_levels = [0.683, 0.954]

# Plot variables
x_key = "mu"
y_key = "sigma"
z_key = "LogLike"

# Set some bounds manually?
dataset_bounds = {
    "mu": [19.5, 22.0],
    "sigma": [1.0, 4.0],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": r"$\mu$ (unit)",
    "sigma": r"$\sigma$ (unit)",
    "LogLike": r"$\ln L$",
    "color_data": r"Color data",
}

# Number of bins used for profiling
xy_bins = (20, 20)

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

plot_settings["interpolation"] = True
plot_settings["interpolation_resolution"] = 200

plot_settings["separator_linewidth"] = 2 * plot_settings["framewidth"]
plot_settings["separator_color"] = "white"

plot_settings["max_likelihood_marker"] = "D"  # Diamond marker
plot_settings["max_likelihood_marker_size"] = 10
plot_settings["max_likelihood_marker_linewidth"] = 0.8

plot_settings["contour_linewidth"] = 1.0
plot_settings["contour_color"] = "white"
plot_settings["contour_linestyle"] = "solid"

plot_settings["1D_profile_likelihood_color"] = "crimson"
plot_settings["1D_profile_likelihood_fill_alpha"] = 0.3


# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)
y_label = plot_labels.get(y_key, y_key)
z_label = plot_labels.get(z_key, z_key) 

labels = (x_label, y_label, z_label)

fig, ax, cbar_ax = plot_utils.plot_conditional_profile_intervals(
    data[x_key], 
    data[y_key], 
    data[z_key], 
    labels, 
    xy_bins, 
    xy_bounds=xy_bounds, 
    z_bounds=None, 
    confidence_levels=confidence_levels,
    draw_interval_connectors=False,
    add_max_likelihood_marker=True,
    shaded_confidence_interval_bands=True,
    x_condition="bin",  # "bin", "upperbound", "lowerbound"
    plot_settings=plot_settings,
)

# Add header
header_text = r"Conditional $1\sigma$ and $2\sigma$ CL intervals."
if plt.rcParams.get("text.usetex"):
    header_text += r" \textsf{GAMBIT} 2.5"
else:
    header_text += r" GAMBIT 2.5"
add_header(header_text, ax=ax)

# Save to file
output_path = f"./plots/conditional_profile_intervals__{x_key}__{y_key}__{z_key}__shaded.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")
