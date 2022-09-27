"""
    This module can be used to generate error files for generated graph types.
    Extension of the GenerateErrorFiles module. Can be merged in the future.
"""

from typing import List, Optional, Tuple, Dict
from os import listdir
from os.path import isfile, join
import sys
import IO
import os
from pathlib import Path

Project_Path = os.path.dirname(os.path.abspath(__file__))


def run(Graph_Path):
    """
    This function generates the error values from the approximated values and the exact values and saves them to a file.
    Different runs can be processed simultaneously. See naming convention for approximation files.

    Args:
        Graph_Path: the path where exact and approximated values are stored. Make sure there is also an Erros folder.

    Returns:
        Nothing - Error files will be saved locally in the Errors folder
    """
    # Change here the maximal Number of Runs that was used in the previous steps
    max_run = 10

    exact_path = f"{Graph_Path}\\Exact_Betweenness"
    approx_path = f"{Graph_Path}\\Approx_Betweenness"
    error_path = f"{Graph_Path}\\Errors"

    approx_files = [f for f in listdir(approx_path) if isfile(join(approx_path, f))]
    exact_files = [f for f in listdir(exact_path) if isfile(join(exact_path, f))]

    for file in approx_files:
        split_string = file.split("_", 7)
        print(split_string)

        # Get all normalized files and the according exact values
        if split_string[0] == "Approx" and split_string[7] == "norm_True.txt":
            found = False
            for run in range(1, max_run + 1):
                exact_file_name = f"Exact_{split_string[1]}_{split_string[2]}_{split_string[3]}_{split_string[4]}_{run}_Abs_norm_True.txt"  # Change here run and
                if exact_file_name in exact_files:
                    exact = IO.file_to_dict(f"{exact_path}\\{exact_file_name}")
                    approx = IO.file_to_dict(f"{approx_path}\\{file}")

                    # Calculate absolute error
                    error = {
                        key: exact[key] - approx.get(key, 0) for key in exact.keys()
                    }

                    # Write Error Files
                    with open(
                        f"{error_path}\\Error_{split_string[1]}_{split_string[2]}_{split_string[3]}_{split_string[4]}_{split_string[5]}_{split_string[6]}.txt",
                        "w",
                    ) as fp:
                        fp.write(
                            "\n".join(
                                "{}:    {}".format(node, x) for node, x in error.items()
                            )
                        )
                    print(
                        f"Generated {error_path}\\Error_{split_string[1]}_{split_string[2]}_{split_string[3]}_{split_string[4]}_{split_string[5]}_{split_string[6]}.txt"
                    )
                    found = True
            if not found:
                print(f"{exact_file_name} seems to be missing")
    return


def main():
    """
    In this function one can select the different graph generation types and the according
    path will be handed to the function which generated the error files
    """
    # TODO:
    # - Selection as arguments
    # - reconstruct run function more flexible

    # Select for which graphs you want to run approximations and exact values
    Configurationmodel_nconstant = True
    Configurationmodel_dconstant = True
    ErdosRenyi = True

    if Configurationmodel_nconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant n"
        )
        run(Path(f"{Project_Path}\\Graphs Generated\\Configuration Model\\n constant"))

    if Configurationmodel_dconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant degree"
        )
        run(Path(f"{Project_Path}\\Graphs Generated\\Configuration Model\\d constant"))

    if ErdosRenyi:
        print("Started to run Approximations for Erdos Renyi Graphs")
        run(Path(f"{Project_Path}\\Graphs Generated\\Erdos Renyi"))


if __name__ == "__main__":
    sys.exit(main())
