from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import matplotlib
import sys


import gambit_plotting_tools.gambit_plot_utils as plot_utils
import gambit_plotting_tools.gambit_plot_settings as gambit_plot_settings




# 
# Read file
# arguments:
# number name likelihood measure

hdf5_file_and_group_names = [
    ("./example_data/results_multinest.hdf5" , "data"),
]

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("Posterior", ("Posterior", float)),
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
# Make  posterior corner plots
# 

credible_regions = [0.683, 0.954]

# Make a corner plot of the parameters in x_keys

x_keys = [
    "mu", 
    "log_mu",
    "sigma", 
    "sigma2",
]

posterior_weights_key = "Posterior"

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

# dimension of the plot:
dim = len(x_keys)

# output path of corner plot
output_path = f"./plots/corner_posterior.pdf"

# make figure
fig1 = plt.figure(figsize=(10,10))

# grid divider (required to get a colorbar below the plots)
gs = gridspec.GridSpec(dim+1, dim, height_ratios=np.ones(dim).tolist()+[0.1], hspace=0.4, wspace=0.1)


for x, x_key in enumerate(x_keys):
    for y, y_key in enumerate(x_keys):

        # If variable bounds are not specified in dataset_bounds, use the full range from the data
        x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
        y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
        xy_bounds = (x_bounds, y_bounds)

        # If a pretty plot label is not given, just use the key
        x_label = plot_labels.get(x_key, x_key)
        y_label = plot_labels.get(y_key, y_key)
        labels = (x_label, y_label)

        # Load default plot settings (and make adjustments if necessary)
        plot_settings = deepcopy(gambit_plot_settings.plot_settings)
        

        # make empty slots above the diagonal
        if x > y:
            ax1 = fig1.add_subplot(gs[y, x])
            ax1.axis('off')

        # add 1D plots along diagonal
        elif y == x:
            plot_settings["1D_posterior_color"] = "purple"
            ax1 = fig1.add_subplot(gs[y, x])

            fig, ax = plot_utils.plot_1D_posterior(
                data[x_key], 
                data[posterior_weights_key], 
                x_label, 
                x_bins, 
                ax=ax1,   # specify axis
                fig=fig1, # specify figure
                x_bounds=x_bounds,
                credible_regions=credible_regions,
                plot_relative_probability=True,
                add_mean_posterior_marker=True,
                fill_color_below_graph=False,
                shaded_credible_region_bands=True,
                plot_settings=plot_settings,
            )
            

        # add 2D plots below the diagonal
        else:
            plot_settings["interpolation"] = "none"
            plot_settings["colormap"] = matplotlib.colormaps["inferno"]

            ax1 = fig1.add_subplot(gs[y, x])
            fig, ax, cbar_ax = plot_utils.plot_2D_posterior(
                data[x_key], 
                data[y_key], 
                data[posterior_weights_key], 
                labels, 
                xy_bins, 
                ax=ax1,   # specify axis
                fig=fig1, # specify figure
                gs=gs,    # specify gridspec
                corner_colorbar=corner_plot,
                xy_bounds=xy_bounds,
                credible_regions=credible_regions,
                plot_relative_probability=True,
                add_mean_posterior_marker=True,
                plot_settings=plot_settings, # specify that you want a corner plot with only one colorbar
            )
            
            # Add a star marker at the maximum likelihood point
            max_like_index = np.argmax(data["LogLike"])
            x_max_like = data[x_key][max_like_index]
            y_max_like = data[y_key][max_like_index]
            ax1.scatter(x_max_like, y_max_like, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
                    edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100)


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





