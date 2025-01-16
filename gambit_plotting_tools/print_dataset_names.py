"""
Print data set names
====================
"""

import argparse
import h5py


def main():

    parser = argparse.ArgumentParser(
        description="Read a GAMBIT hdf5 file and print all the dataset names.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        print_dataset_names path/to/my_file.hdf5 --ignore-endswith \"_isvalid\" --ignore-startswith \"metadata\"
        print_dataset_names path/to/my_file.hdf5 --include-contains \"primary_parameters\"
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
        help="List of strings. Datasets with names beginning with a string in this list will not be printed."
    )
    parser.add_argument(
        "--ignore-endswith",
        type=str,
        nargs='*',
        default=[],
        help="List of strings. Datasets with names ending on a string in this list will not be printed."
    )
    parser.add_argument(
        "--ignore-contains",
        type=str,
        nargs='*',
        default=[],
        help="List of strings. Datasets with names containing a string in this list will not be printed."
    )
    parser.add_argument(
        "--include-contains",
        type=str,
        nargs='*',
        default=[],
        help="List of strings. If provided, only datasets whose names contain a string in this list will be printed."
    )

    args = parser.parse_args()

    names = get_dataset_names(args.file_name, args.ignore_startswith,
                              args.ignore_endswith, args.ignore_contains, args.include_contains)

    for n in names:
        print(n)


def get_dataset_names(file_name, ignore_startswith, ignore_endswith, ignore_contains, include_contains):

    dataset_names = []

    def collect_datasets(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        for ignored_start in ignore_startswith:
            if name.startswith(ignored_start):
                return

        for ignored_end in ignore_endswith:
            if name.endswith(ignored_end):
                return

        for ignored_substring in ignore_contains:
            if ignored_substring in name:
                return

        for included_substring in include_contains:
            if included_substring not in name:
                return

        dataset_names.append(name)

    with h5py.File(file_name, 'r') as file:
        # Visit all datasets in the group and collect the dataset names
        file.visititems(collect_datasets)

    return dataset_names


if __name__ == "__main__":
    main()
