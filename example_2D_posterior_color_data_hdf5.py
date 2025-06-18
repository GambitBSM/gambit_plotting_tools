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

hdf5_file = "./example_data/results_multinest.hdf5" 
group_name = "data"

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
datasets = [
    ("LogLike", ("LogLike", float)),
    ("Posterior", ("Posterior", float)),
    ("mu",      ("#NormalDist_parameters @NormalDist::primary_parameters::mu", float)),
    ("sigma",   ("#NormalDist_parameters @NormalDist::primary_parameters::sigma", float)),
]

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([(hdf5_file, group_name)], datasets, filter_invalid_points=True)

# Create a dummy dataset to demonstrate the color_data feature
data["color_data"] = np.sin(data["mu"] / 10) * np.cos(data["sigma"] / 2)

# 
# Make a 2D posterior plot
# 

credible_regions = [0.683, 0.954]

# Plot variables
x_key = "mu"
y_key = "sigma"
posterior_weights_key = "Posterior"
c_key = "color_data"

# Set some bounds manually?
dataset_bounds = {
    "mu": [15, 30],
    "sigma": [0, 5],
    "color_data": [-1, 1],
}

# Specify some pretty plot labels?
plot_labels = {
    "mu": r"$\mu$ (unit)",
    "sigma": r"$\sigma$ (unit)",
    "color_data": r"$\sin(\mu/10) \cos(\sigma/2)$"
}

# Number of bins
xy_bins = (100, 100)

# Load default plot settings (and make adjustments if necessary)
plot_settings = deepcopy(gambit_plot_settings.plot_settings)
plot_settings["colormap"] = matplotlib.colormaps["managua"]

# Discretize colormap?
n_colors = 20
plot_settings["colormap"] = ListedColormap(plot_settings["colormap"](np.linspace(0, 1, n_colors)))

# If variable bounds are not specified in dataset_bounds, use the full range from the data
x_bounds = dataset_bounds.get(x_key, [np.min(data[x_key]), np.max(data[x_key])])
y_bounds = dataset_bounds.get(y_key, [np.min(data[y_key]), np.max(data[y_key])])
xy_bounds = (x_bounds, y_bounds)

# If a pretty plot label is not given, just use the key
x_label = plot_labels.get(x_key, x_key)
y_label = plot_labels.get(y_key, y_key)
labels = (x_label, y_label)

# Create 2D posterior figure
fig, ax, cbar_ax = plot_utils.plot_2D_posterior(
    data[x_key], 
    data[y_key], 
    data[posterior_weights_key], 
    labels, 
    xy_bins, 
    xy_bounds=xy_bounds,
    credible_regions=credible_regions,
    # contour_coordinates_output_file=f"./plots/2D_posterior__{x_key}__{y_key}__coordinates.csv",
    plot_relative_probability=True,
    add_mean_posterior_marker=True,
    color_data=data[c_key],
    color_point_estimate="maxpost", # "mean" or "maxpost"
    color_posterior_bins=n_colors,
    color_label=plot_labels.get(c_key, c_key),
    color_bounds=dataset_bounds["color_data"],
    color_within_credible_region=credible_regions[-1],
    missing_value_color=plot_settings["facecolor_plot"],
    plot_settings=plot_settings,
)

# Add text
fig.text(0.50, 0.30, "Example text", ha="left", va="center", fontsize=plot_settings["fontsize"], color="white")

# Add header
header_text = r"$1\sigma$ and $2\sigma$ credible regions."
if plt.rcParams.get("text.usetex"):
    header_text += r" \textsf{GAMBIT} 2.5"
else:
    header_text += r" GAMBIT 2.5"
add_header(header_text, ax=ax)

# Add anything else to the plot, e.g. some more lines and labels and stuff
ax.plot([20.0, 30.0], [5.0, 3.0], color="white", linewidth=plot_settings["contour_linewidths"][0], linestyle="dashed")
fig.text(0.53, 0.79, "A very important line!", ha="left", va="center", rotation=-31.5, fontsize=plot_settings["fontsize"]-5, color="white")

# Add a star marker at the maximum likelihood point
max_like_index = np.argmax(data["LogLike"])
x_max_like = data[x_key][max_like_index]
y_max_like = data[y_key][max_like_index]
ax.scatter(x_max_like, y_max_like, marker=plot_settings["max_likelihood_marker"], s=plot_settings["max_likelihood_marker_size"], c=plot_settings["max_likelihood_marker_color"],
           edgecolor=plot_settings["max_likelihood_marker_edgecolor"], linewidth=plot_settings["max_likelihood_marker_linewidth"], zorder=100)

# Save to file
output_path = f"./plots/2D_posterior__{x_key}__{y_key}__{c_key}.pdf"
plot_utils.create_folders_if_not_exist(output_path)
plt.savefig(output_path)
plt.close()
print(f"Wrote file: {output_path}")



