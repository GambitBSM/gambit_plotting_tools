from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_header
from gambit_plotting_tools.gambit_colormaps import register_cmaps


# Set styling
register_cmaps()
plt.style.use(['gambit_plotting_tools.gambit', 'gambit_plotting_tools.dark'])

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

confidence_levels = [0.683, 0.954]
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

# Plot variables
x_key = "mu"
y_key = "sigma"
z_key = "LogLike"

# Set some bounds manually?
dataset_bounds = {
    "mu": [15, 30],
    "sigma": [0, 5],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": "$\\mu$ (unit)",
    "sigma": "$\\sigma$ (unit)",
}

# Number of bins used for profiling
xy_bins = (100, 100)

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)
y_label = plot_labels.get(y_key, y_key)

# Create 2D profile likelihood figure
fig, ax, cbar_ax = plot_utils.plot_2D_profile(
    data[x_key], 
    data[y_key], 
    data[z_key], 
    xy_bins, 
    xy_bounds=xy_bounds, 
    z_is_loglike=True,
    plot_likelihood_ratio=True,
    contour_levels=likelihood_ratio_contour_values,
    #contour_coordinates_output_file=f"./plots/2D_profile__{x_key}__{y_key}__{z_key}__coordinates.csv",
    z_fill_value = -1*np.finfo(float).max,
    add_max_likelihood_marker = True,
    plot_settings=plot_settings,
)

# Set limits and labels
ax.set_xlim(*x_bounds)
ax.set_ylim(*y_bounds)

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

# Add text
fig.text(0.525, 0.350, "Example text", ha="left", va="center")

# Add header
header_text = "$1\\sigma$ and $2\\sigma$ CL regions. \\textsf{GAMBIT} 2.5"
add_header(header_text, ax=ax)

# Add anything else to the plot, e.g. some more lines and labels and stuff
ax.plot([20.0, 30.0], [5.0, 3.0], linestyle="dashed")
fig.text(0.53, 0.79, "A very important line!", ha="left", va="center", rotation=-31.5, fontsize="x-small")

# Draw a contour using coordinates stored in a .csv file
x_contour, y_contour = np.loadtxt("./example_data/contour_coordinates.csv", delimiter=",", usecols=(0, 1), unpack=True)
ax.plot(x_contour, y_contour, color="orange", linestyle="dashed", linewidth=plot_settings["contour_linewidth"], alpha=0.7)
fig.text(0.23, 0.23, "Overlaid contour from\n coordinates in some data file", ha="left", va="center", fontsize="x-small", color="orange")

# Save to file
output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")



