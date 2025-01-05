#!/usr/bin/env python3

import argparse
import numpy as np
import h5py

import gambit_plotting_tools.gambit_plot_utils as plot_utils


#
# Parse input arguments
#

parser = argparse.ArgumentParser(
    description="Read a GAMBIT hdf5 file and print the parameters of the points with highest log-likelihood.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog="""Example usage:
    python print_high_loglike_points.py path/to/my_file.hdf5 data 3
    """
)

parser.add_argument("file_name", type=str, help="Input hdf5 file name.")
parser.add_argument("group_name", type=str, help="Name of the hdf5 group in the input file.")
parser.add_argument("print_n_points", type=int, help="The number of points to print.")

args = parser.parse_args()

file_name = args.file_name
group_name = args.group_name
print_n_points = args.print_n_points


# 
# Collect info and read file
#

# Check if the hdf5 file was created by GAMBIT or GAMBIT-light
gambit_version = None
with h5py.File(file_name, 'r') as file:
    if "metadata" in file:
        if "GAMBIT" in file["metadata"]:
            gambit_version = "gambit"
        elif "GAMBIT-light" in file["metadata"]:
            gambit_version = "gambit-light"

if gambit_version is None:
    raise Exception("Cannot find the dataset 'metadata/GAMBIT' or 'metadata/GAMBIT-light'.")

if gambit_version == "gambit-light":
    raise Exception("This script currently only works with hdf5 files created with GAMBIT, not with GAMBIT-light.")


hdf5_file_and_group_name = (file_name, group_name)

# Get all dataset names
all_dataset_names = plot_utils.collect_all_dataset_names(hdf5_file_and_group_name)

# Get all model names
all_model_names = plot_utils.collect_all_model_names(hdf5_file_and_group_name)

# Get dict from <model name> to <list of parameter names> for all models and parameters
model_param_dict = plot_utils.collect_all_model_and_param_names(hdf5_file_and_group_name)

# Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
# to specify all datasets to be read from the file
load_datasets = []
for dset_name in all_dataset_names:

    if dset_name == "LogLike":
        load_datasets.append(("LogLike", ("LogLike", float)))

    if "::primary_parameters::" in dset_name:
        short_param_name = dset_name.split("::")[-1]
        load_datasets.append((short_param_name, (dset_name, float)))        

# Now create our main data dictionary by reading the hdf5 files
data = plot_utils.read_hdf5_datasets([hdf5_file_and_group_name], load_datasets, filter_invalid_points=True, verbose=False)


#
# Print parameters of the highest loglike points
#

# Reorder all datasets so that points with high log-likelihood appear first
p = np.argsort(data['LogLike'])[::-1]
for key in data.keys():
    data[key] = data[key][p]

# Finally, print parameter points to screen (in yaml format):
print()
print(f"File:  {file_name}")
print(f"Group: {group_name}")
print()
print(f"The {print_n_points} highest log-likelihood point(s):")
print()
for i in range(print_n_points):
    loglike_val = data["LogLike"][i]
    print(f"LogLike: {loglike_val:.14e}")
    print("Parameters:")
    for model_name in all_model_names:
        print(f"  {model_name}:")
        for param_name in model_param_dict[model_name]:
            param_val = data[param_name][i]
            print(f"    {param_name}: {param_val:.14e}")
    print()
    print()
