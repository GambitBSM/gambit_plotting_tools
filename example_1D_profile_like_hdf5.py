from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import plot_utils
import gambit_plot_settings


# 
# Read file
# 

hdf5_file = "./example_data/samples_run1.hdf5" 
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)



# 
# Make a 1D profile likelihood plot
# 

confidence_levels = [0.683, 0.954]
# confidence_levels = [0.683]
# likelihood_ratio_contour_values = plot_utils.get_1D_likelihood_ratio_levels(confidence_levels)

# Plot variables
x_key = "sigma"
y_key = "LogLike"

# Set some bounds manually?
dataset_bounds = {
    "sigma": [0, 5],
    # "sigma": [0, 2.5],
}

# Specify some pretty plot labels?
plot_labels = {
    "sigma": "$\\sigma$ (unit)",
}

# Number of bins used for profiling
x_bins = 100

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)

# Create 1D profile likelihood figure
fig, ax = plot_utils.plot_1D_profile(
    data[x_key], 
    data[y_key], 
    x_label, 
    x_bins, 
    x_bounds=x_bounds, 
    y_is_loglike=True,
    plot_likelihood_ratio=True,
    confidence_levels=confidence_levels,
    y_fill_value = -1*np.finfo(float).max,
    add_max_likelihood_marker = True,
    fill_color_below_graph=False,
    shaded_confidence_interval_bands=True,
    plot_settings=plot_settings,
)

# Add text
header_text = "\\textsf{GAMBIT} 2.5"
fig.text(1.0-0.18, 1.0-0.05, header_text, ha="right", va="bottom", fontsize=plot_settings["header_fontsize"])
fig.text(0.53, 0.85, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="black")

# Save to file
output_path = f"./plots/1D_profile__{x_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")



