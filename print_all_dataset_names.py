import argparse
import numpy as np
import h5py


#
# Parse input arguments
#

parser = argparse.ArgumentParser(
    description="Read a GAMBIT hdf5 file and print all the dataset names.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog="""Example usage:
    python print_all_dataset_names.py path/to/my_file.hdf5 --ignore-endswith \"_isvalid\" --ignore-startswith \"metadata\"
    """
)

parser.add_argument(
    "file_name", 
    type=str, 
    help="Input hdf5 file name."
)
parser.add_argument(
    "--ignore-startswith",
    type=str, 
    nargs='*', 
    default=[], 
    help="List of strings. Datasets with names beginning with a string in this list will not be printed. If not provided, defaults to an empty list."
)
parser.add_argument(
    "--ignore-endswith",
    type=str, 
    nargs='*', 
    default=[], 
    help="List of strings. Datasets with names ending on a string in this list will not be printed. If not provided, defaults to an empty list."
)
parser.add_argument(
    "--ignore-contains",
    type=str, 
    nargs='*', 
    default=[], 
    help="List of strings. Datasets with names containing a string in this list will not be printed. If not provided, defaults to an empty list."
)

args = parser.parse_args()

file_name = args.file_name
ignore_startswith = args.ignore_startswith
ignore_endswith = args.ignore_endswith
ignore_contains = args.ignore_contains


#
# Print dataset names
#

dataset_names = []
with h5py.File(file_name, 'r') as file:

    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            skip = False
            for ignored_start in ignore_startswith:
                if name.startswith(ignored_start):
                    skip = True
                    break
            if not skip:
                for ignored_end in ignore_endswith:
                    if name.endswith(ignored_end):
                        skip = True
                        break
            if not skip:
                for ignored_substring in ignore_contains:
                    if ignored_substring in name:
                        skip = True
                        break

            if not skip:
                dataset_names.append(name)
    
    # Visit all datasets in the group and collect the dataset names
    file.visititems(collect_datasets)

for dset_name in dataset_names:
    print(f"{dset_name}")

