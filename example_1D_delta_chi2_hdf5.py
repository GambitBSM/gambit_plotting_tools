from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings
from gambit_plotting_tools.annotate import add_gambit_header, add_gambit_logo

#
# Read file
#

hdf5_file = "./example_data/results_run1.hdf5"
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)

# Convert log-likelihood to chi^2
data["chi2"] = -2 * data["LogLike"]


#
# Make a 1D delta chi^2 plot
#

confidence_levels = [0.683, 0.954]

# Plot variables
x_key = "sigma"
y_key = "chi2"

# Set some bounds manually?
dataset_bounds = {
    "sigma": [0, 5],
}

# Specify some pretty plot labels?
plot_labels = {
    "sigma": r"$\sigma$ (unit)",
}

# Number of bins used for profiling
x_bins = 100

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)

# Create 1D delta chi^2 figure
fig, ax = plot_utils.plot_1D_delta_chi2(
    data[x_key],
    data[y_key],
    x_label,
    x_bins,
    x_bounds=x_bounds,
    y_data_is_chi2=True,
    confidence_levels=confidence_levels,
    y_fill_value = np.finfo(float).max,
    add_best_fit_marker = True,
    fill_color_below_graph=False,
    shaded_confidence_interval_bands=True,
    plot_settings=plot_settings,
    graph_coordinates_output_file=f"./plots/1D_delta_chi2__{x_key}__{y_key}__coordinates.csv"
)

# Add text
fig.text(0.53, 0.85, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="black")

# Add branding
add_gambit_header(ax=ax, version="2.5")
add_gambit_logo(ax=ax)

# Save to file
output_path = f"./plots/1D_delta_chi2__{x_key}__{y_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")


