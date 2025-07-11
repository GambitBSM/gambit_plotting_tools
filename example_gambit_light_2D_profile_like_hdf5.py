from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_header


# 
# Read file
# 

hdf5_file = "path/to/your/gambit_light/runs/gambit_light_example_rosenbrock_scan/samples/results.hdf5" 
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("x1",      ("input::x1", float)),
    ("x2",      ("input::x2", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)



# 
# Make a 2D profile likelihood plot
# 

confidence_levels = [0.683, 0.954]
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

# Plot variables
x_key = "x1"
y_key = "x2"
z_key = "LogLike"

# Set some bounds manually?
dataset_bounds = {
    "x1": [-5, 10],
    "x2": [-5, 10],
}

# Specify some pretty plot labels?
plot_labels = {
    "x1": r"$x_{1}$ (unit)",
    "x2": r"$x_{2}$ (unit)",
}

# Number of bins used for profiling
xy_bins = (200, 200)

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)
y_label = plot_labels.get(y_key, y_key)
z_label = plot_labels.get(z_key, z_key)
labels = (x_label, y_label, z_label)

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
    missing_value_color=plot_settings["colormap"](0.0),
    add_max_likelihood_marker = True,
    plot_settings=plot_settings,
)

# Add header
header_text = r"$1\sigma$ and $2\sigma$ CL regions."
add_header(header_text, ax=ax)

# Save to file
output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")



