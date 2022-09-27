"""
This module was constructed to streamline the execution of the most important functions/modules
"""

import RunApproximations
import GenerateErrorFiles
import PlottingLibary

import sys


def main(graphs):
    """
    Runs approximations, generates the related error files and then visualizes the errors against other parameters

    Args:
        graphs: list of graphs which should be used

    Returns:
        None
    """
    # Run important Functions
    RunApproximations.main(graphs)
    GenerateErrorFiles.main(graphs)
    PlottingLibary.main(graphs)


if __name__ == "__main__":
    sys.exit(main(["ca-GrQc", "ca-HepPh", "ca-HepTh"]))
