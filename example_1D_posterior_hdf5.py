from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import plot_utils
import gambit_plot_settings


# 
# Read file
# 

hdf5_file = "./example_data/samples_multinest.hdf5" 
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("Posterior", ("Posterior", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)



# 
# Make a 1D posterior plot
# 

credible_regions = [0.683, 0.954]

# Plot variables
x_key = "sigma"
posterior_weights_key = "Posterior"

# Set some bounds manually?
dataset_bounds = {
    "sigma": [0, 5],
}

# Specify some pretty plot labels?
plot_labels = {
    "sigma": "$\\sigma$ (unit)",
}

# Number of bins
x_bins = 80

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)

# Create 1D posterior figure
fig, ax = plot_utils.plot_1d_posterior(
    data[x_key], 
    data[posterior_weights_key], 
    x_label, 
    x_bins, 
    x_bounds=x_bounds,
    credible_regions=credible_regions,
    plot_relative_probability=True,
    add_mean_posterior_marker=True,
    plot_settings=plot_settings,
)

# Add text
header_text = "\\textsf{GAMBIT} 2.5"
fig.text(1.0-0.18, 1.0-0.05, header_text, ha="right", va="bottom", fontsize=plot_settings["header_fontsize"])
fig.text(0.53, 0.85, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="black")

# Add a star marker at the maximum likelihood point
max_like_index = np.argmax(data["LogLike"])
x_max_like = data[x_key][max_like_index]
y_max_like = 0.0
ax.scatter(x_max_like, y_max_like, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
           edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100, clip_on=False)

# Save to file
output_path = f"./plots/1D_posterior__{x_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")



