"""
Get high likelihood points from data sets
=========================================
"""


import argparse
import numpy as np
import h5py

from . import gambit_plot_utils as plot_utils


LOGLIKE = "LogLike"


def is_gambit_light(file_name):
    """
    @returns Whether file is from gambit light
    """
    with h5py.File(file_name, 'r') as file:
        if "metadata" in file:
            if "GAMBIT" in file["metadata"]:
                return False
            if "GAMBIT-light" in file["metadata"]:
                return True

    raise RuntimeError(
        "Cannot find the dataset 'metadata/GAMBIT' or 'metadata/GAMBIT-light'")


def is_dataset(name):
    """
    @returns Whether entry should be considered data
    """
    return name == LOGLIKE or "::primary_parameters::" in name


def dataset_type(name):
    """
    @returns HDF5-style type information
    """
    short_name = name.split("::")[-1]
    return (short_name, (name, float))


def get_high_likelihod(data, n_points):
    """
    @returns High likelihood points from data from data
    """
    idx = np.argsort(data[LOGLIKE])[-n_points:]
    return [{k: data[k][i] for k in data} for i in idx]


def read_high_likelihood(file_name, group_name, n_points):
    """
    @returns High likelihood points from file
    """

    # Check if the hdf5 file was created by GAMBIT or GAMBIT-light
    if is_gambit_light(file_name):
        raise RuntimeError(
            "This script currently only works with hdf5 files created with GAMBIT, not with GAMBIT-light.")

    hdf5_file_and_group_name = (file_name, group_name)

    # Get all dataset names
    dataset_names = plot_utils.collect_all_dataset_names(
        hdf5_file_and_group_name)

    # Create a list of tuples of the form (shorthand key, (full dataset name, dataset type))
    # to specify all datasets to be read from the file
    load_datasets = [dataset_type(name)
                     for name in dataset_names if is_dataset(name)]

    # Now create our main data dictionary by reading the hdf5 files
    data = plot_utils.read_hdf5_datasets(
        [hdf5_file_and_group_name], load_datasets, filter_invalid_points=True, verbose=False)

    # Select high-likelihood points
    return get_high_likelihod(data, n_points)


def to_yaml(model_param_dict, point):
    """
    @returns Point in YAML format as string
    """
    yaml = f"{LOGLIKE}: {point[LOGLIKE]:.14e}\n"
    yaml += "Parameters:\n"
    for model in model_param_dict:
        yaml += f"  {model}:\n"
        for param in model_param_dict[model]:
            yaml += f"    {param}: {point[param]:.14e}\n"
    return yaml


def main():

    parser = argparse.ArgumentParser(
        description="Read a GAMBIT hdf5 file and print the parameters of the points with highest log-likelihood.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        print_high_loglike_points path/to/my_file.hdf5 data 3
        """
    )

    parser.add_argument("file_name", type=str, help="Input hdf5 file name.")
    parser.add_argument("group_name", type=str,
                        help="Name of the hdf5 group in the input file.")
    parser.add_argument("n_points", type=int,
                        help="The number of points to print.")

    args = parser.parse_args()

    # Read points

    points = read_high_likelihood(
        args.file_name, args.group_name, args.n_points)
    model_param_dict = plot_utils.collect_all_model_and_param_names(
        (args.file_name, args.group_name))

    # Print in yaml format

    print()
    print(f"File:  {args.file_name}")
    print(f"Group: {args.group_name}")
    print()
    print(f"The {args.n_points} highest log-likelihood point(s):")
    print()
    for p in points:
        print(to_yaml(model_param_dict, p))


if __name__ == "__main__":
    main()
