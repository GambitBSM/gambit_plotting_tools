from copy import deepcopy
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import plot_utils
import gambit_plot_settings



# 
# Read data file
# 

data_file = "./example_data/samples.dat"

# Create a dictionary that maps a shorthand key to the datafile column number
datasets = OrderedDict([
    ("p1", 0),
    ("p2", 1),
    ("p3", 2),
    ("lnL", 3),
])

# Now create our main data dictionary by reading the ascii file
data_arr = np.loadtxt(data_file, comments="#", usecols=datasets.values(), unpack=False)
data = {key: data_arr[:,i] for i,key in enumerate(datasets.keys())}



# 
# Make a 2D profile likelihood plot
# 

confidence_levels = [0.683, 0.954]
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

# Plot variables
x_key = "p1"
y_key = "p2"
z_key = "lnL"

# Set some bounds manually?
dataset_bounds = {
    "p1": [0.0, 1.0],
    "p2": [0.1, 0.7],
}

# Specify some pretty plot labels?
plot_labels = {
    "p1": "$p_1$ (unit)",
    "p2": "$p_2$ (unit)",
    "p3": "$p_3$ (unit)",
    "lnL": "$\\ln(L)$",
}

# Number of bins used for profiling
xy_bins = (70, 70)

# Load default plot settings and make some adjustments
plot_settings = deepcopy(gambit_plot_settings.plot_settings)
plot_settings["colormap"] = matplotlib.colormaps["plasma"]
plot_settings["interpolation"] = "none"

# If variable bounds are not specified, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

labels = (plot_labels[x_key], plot_labels[y_key], plot_labels[z_key])

# Create 2D profile likelihood figure
fig, ax, cbar_ax = plot_utils.plot_2D_profile(
    data[x_key], 
    data[y_key], 
    data[z_key], 
    labels,
    xy_bins, 
    xy_bounds=xy_bounds, 
    z_is_loglike=True,
    plot_likelihood_ratio=True,
    contour_levels=likelihood_ratio_contour_values,
    z_fill_value = -1*np.finfo(float).max,
    add_max_likelihood_marker = True,
    plot_settings=plot_settings,
)

# Add text
header_text = "$1\\sigma$ and $2\\sigma$ CL regions. \\textsf{GAMBIT} 2.5"
fig.text(1.0-0.18, 1.0-0.05, header_text, ha="right", va="bottom", fontsize=plot_settings["header_fontsize"])
fig.text(0.27, 0.83, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="white")

# Add anything else to the plot, e.g. some more lines and labels and stuff
ax.plot([0.0, 1.0], [0.0, 1.0], color="white", linewidth=plot_settings["contour_linewidth"], linestyle="dashed")
fig.text(0.25, 0.35, "A very important line!", ha="left", va="center", rotation=59.5, fontsize=plot_settings["fontsize"]-5, color="white")

# Save to file
output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
print(f"Wrote file: {output_path}")
