from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import plot_utils
import gambit_plot_settings


# 
# Read files
# 

hdf5_file_and_group_names = [
    ("./example_data/samples_run1.hdf5", "data"),
    ("./example_data/samples_run2.hdf5", "data"),
]

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("mu",      ("#NormalDist_parameters @NormalDist::primary_parameters::mu", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets(hdf5_file_and_group_names, datasets, filter_invalid_points=True)


#
# Add any derived datasets we might be interested in 
#

data["sigma2"] = data["sigma"]**2
data["log_mu"] = np.log(data["mu"])


# 
# Make 2D profile likelihood plots
# 

confidence_levels = [0.683, 0.954]
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

# Make a plot for every combination of (x_key, y_key, z_key)

x_keys = [
    "mu", 
    "log_mu",
]

y_keys = [
    "sigma", 
    "sigma2"
]

z_keys = [
    "LogLike",
]

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

for z_key in z_keys:
    for x_key in x_keys:
        for y_key in y_keys:

            # If a pretty plot label is not given, just use the key
            x_label = plot_labels.get(x_key, x_key)
            y_label = plot_labels.get(y_key, y_key)
            z_label = plot_labels.get(z_key, z_key)
            labels = (x_label, y_label, z_label)

            # If variable bounds are not specified, use the full range from the data
            x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
            y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
            xy_bounds = (x_bounds, y_bounds)

            # Copy default GAMBIT plot settings (and make changes if necessary)
            plot_settings = deepcopy(gambit_plot_settings.plot_settings)

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

            # Save to file
            output_path = f"./plots/2D_profile__{x_key}__{y_key}__{z_key}.pdf"
            plot_utils.create_folders_if_not_exist(output_path)
            plt.savefig(output_path)
            plt.close()
            print(f"Wrote file: {output_path}")



