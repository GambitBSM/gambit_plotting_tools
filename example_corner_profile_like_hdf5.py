from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import sys


import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings




# 
# Read file
# arguments:
# number name likelihood measure

hdf5_file_and_group_names = [
    ("./example_data/results_run1.hdf5", "data"),
    ("./example_data/results_run2.hdf5", "data"),
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
# Nearest Neighbor averaging
#
k = 30
NN_instance = NearestNeighbors(n_neighbors=k, algorithm='auto')
scaler = StandardScaler()

data["LogLike_avg"] = plot_utils.nearest_neighbor_averaging(
    hdf5_file_and_group_names, 
    "LogLike", 
    NN_instance,
    parameter_dataset_tag="::primary_parameters::", 
    scaler=scaler, 
    filter_invalid_points=True
)

#
# Add any derived datasets we might be interested in 
#

data["sigma2"] = data["sigma"]**2
data["log_mu"] = np.log(data["mu"])


# 
# Make profile likelihood corner plots
# 

confidence_levels = [0.683, 0.954]
likelihood_ratio_contour_values = plot_utils.get_2D_likelihood_ratio_levels(confidence_levels)

# Make a corner plot of the parameters in x_keys, with different likelihoods z_keys

x_keys = [
    "mu", 
    "log_mu",
    "sigma", 
    "sigma2",
]

z_keys = [
    "LogLike",
    "LogLike_avg"
]

# Set some bounds manually?
dataset_bounds = {
    "mu": [15, 30],
    "log_mu": [2.7, 3.4],
    "sigma": [0, 5],
    "sigma2": [0, 25],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": "$\\mu$ (unit)",
    "log_mu": "$\\log(\\mu)$ (unit)",
    "sigma": "$\\sigma$ (unit)",
    "sigma2": "$\\sigma_{2}$ (unit)",
}

# Number of bins used for profiling
xy_bins = (100, 100)
x_bins = 100

# specify that you only want one colorbar at the bottom of the plot
corner_plot = True

# dimension of parameters
dim = len(x_keys)

for z_key in z_keys:

    # output path of corner plot
    output_path = f"./plots/corner_profile_{z_key}.pdf"

    # make figure
    fig1 = plt.figure(figsize=(10,10))

    # grid divider (required to get a colorbar below the plots)
    gs = gridspec.GridSpec(dim+1, dim, height_ratios=np.ones(dim).tolist()+[0.1], hspace=0.5, wspace=0.1)


    for x, x_key in enumerate(x_keys):
        for y, y_key in enumerate(x_keys):

            # If variable bounds are not specified in dataset_bounds, use the full range from the data
            x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
            y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
            xy_bounds = (x_bounds, y_bounds)

            # If a pretty plot label is not given, just use the key
            x_label = plot_labels.get(x_key, x_key)
            y_label = plot_labels.get(y_key, y_key)
            z_label = plot_labels.get(z_key, z_key)
            labels = (x_label, y_label, z_label)

            # Load default plot settings (and make adjustments if necessary)
            plot_settings = deepcopy(gambit_plot_settings.plot_settings)

            # make empty slots above the diagonal
            if x > y:
                ax1 = fig1.add_subplot(gs[y, x])
                ax1.axis('off')

            # add 1D plots along diagonal
            elif y == x:
                ax1 = fig1.add_subplot(gs[y, x])
                plot_utils.plot_1D_profile(
                    data[x_key], 
                    data[z_key], 
                    x_label, 
                    x_bins, 
                    ax=ax1,   # specify axis
                    fig=fig1, # specify figure
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

            # add 2D plots below the diagonal
            else:
                ax1 = fig1.add_subplot(gs[y, x])
                plot_utils.plot_2D_profile(
                    data[x_key], 
                    data[y_key], 
                    data[z_key], 
                    labels, 
                    xy_bins, 
                    ax=ax1,   # specify axis
                    fig=fig1, # specify figure
                    gs=gs,    # specify gridspec
                    xy_bounds=xy_bounds, 
                    z_is_loglike=True,
                    plot_likelihood_ratio=True,
                    contour_levels=likelihood_ratio_contour_values,
                    z_fill_value = -1*np.finfo(float).max,
                    add_max_likelihood_marker = True,
                    plot_settings=plot_settings,
                    corner_colorbar=corner_plot, # specify that you want a corner plot with only one colorbar
                )


            # remove labels and ticks-labels that are not on the outside of the corner plot
            if x != 0:
                ax1.set_yticklabels([])
                ax1.set_ylabel("")
            if y < dim-1:
                ax1.set_xticklabels([])
                ax1.set_xlabel("")

    # save figure and close it
    fig1.savefig(output_path, format='pdf')
    plt.close(fig1)





