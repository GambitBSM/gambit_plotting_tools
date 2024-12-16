import numpy as np
import plot_utils


print_n_points = 1

# Specify input hdf5 file and group name
hdf5_file_and_group_name = ("./example_data/samples_run1.hdf5", "data")


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
data = plot_utils.read_hdf5_datasets([hdf5_file_and_group_name], load_datasets, filter_invalid_points=True)


# Reorder all datasets so that points with high log-likelihood appear first
p = np.argsort(data['LogLike'])[::-1]
for key in data.keys():
    data[key] = data[key][p]


# Finally, print parameter points to screen (in yaml format):
print()
print()
print(f"Listing the {print_n_points} point(s) with highest log-likelihood:")
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
