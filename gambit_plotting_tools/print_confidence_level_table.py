"""
Print confidence level table
============================
"""

import argparse
import scipy.stats
import numpy as np
from tabulate import tabulate


def confidence_level_data(sigma=None, dof=None):
    """
    Adapted from http://www.reid.ai/2012/09/chi-squared-distribution-table-with.html
    """
    if sigma is None:
        sigma = [
            np.sqrt(scipy.stats.chi2.ppf(0.68, 1)),
            1.0,
            np.sqrt(scipy.stats.chi2.ppf(0.9, 1)),
            np.sqrt(scipy.stats.chi2.ppf(0.95, 1)),
            2.0,
            np.sqrt(scipy.stats.chi2.ppf(0.99, 1)),
            3.0,
            np.sqrt(scipy.stats.chi2.ppf(0.999, 1)),
            4.0
        ]

    sigma = np.array(sigma)

    if dof is None:
        dof = np.arange(1, 3)

    dof = np.array(dof)

    # The corresponding confidence levels, in probabilities
    confidence_level = scipy.stats.chi2.cdf(sigma**2, 1)
    p_value = 1. - confidence_level

    chi_squared = [scipy.stats.chi2.ppf(confidence_level, d) for d in dof]
    delta_loglike = [-0.5 * c for c in chi_squared]
    likelihood_ratio = [np.exp(d) for d in delta_loglike]

    # Add strings for tabling
    sigma_str = ["σ"] + sigma.tolist()
    confidence_level_str = ["CL"] + confidence_level.tolist()
    p_value_str = ["p-value"] + p_value.tolist()
    chi_squared_str = [[f"𝛘^2 (dof={d})"] + c.tolist()
                       for c, d in zip(chi_squared, dof)]
    delta_loglike_str = [[f"Δln L (dof={d})"] + dll.tolist()
                         for dll, d in zip(delta_loglike, dof)]
    likelihood_ratio_str = [
        [f"L/L' (dof={d})"] + l.tolist() for l, d in zip(likelihood_ratio, dof)]

    return [sigma_str, confidence_level_str, p_value_str] + chi_squared_str + delta_loglike_str + likelihood_ratio_str


def main():

    # Parse input arguments

    parser = argparse.ArgumentParser(
        description="Print confidence level table.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Example usage:
        print_confidence_level_table
        """
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs='*',
        default=None,
        help="sigma levels to be shown in confidence level table"
    )
    parser.add_argument(
        "--dof",
        type=int,
        nargs='*',
        default=None,
        help="dof to be used in calculations in confidence level table"
    )

    args = parser.parse_args()

    print(tabulate(confidence_level_data(args.sigma, args.dof)))


if __name__ == "__main__":
    main()
