"""
at this point this module consists only of a function which creates the error files
"""

from typing import List, Optional, Tuple, Dict
from os import listdir
from os.path import isfile, join
import sys
import IO
import os


Project_Path = os.path.dirname(os.path.abspath(__file__))


def main(
    graphs: List[str] = [
        "ca-GrQc",
        "email-Enron",
        "ca-HepTh",
        "ca-HepPh",
        "com-amazon",
        "com-lj",
        "dbpedia-link",
    ],
    Original: bool = True,
    path_add: str = "",
    path_add_exact: str = "",
    approx_foldername="Approx_Betweenness",
):
    """
    This function generates the error values from the approximated values and the exact values and saves them to a file

    Args:
        graphs: list of graphs which should be used
        path_add: subfolders (e.g. when using generated graphs)
        path_add_exact: the subfolder of the exact betweenness centrality data
        approx_foldername: the name of the folder where the approximated values are saved (default: "Approx_Betweenness")

        Returns:
            Nothing - Error files will be saved locally in the Errors folder
    """
    if Original:
        exact_path = (
            f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness\\Normalized_Scores"
        )
    else:
        exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness"
    approx_path = f"{Project_Path}\\{path_add}\\{approx_foldername}"
    error_path = f"{Project_Path}\\{path_add}\\Errors"

    approx_files = [f for f in listdir(approx_path) if isfile(join(approx_path, f))]
    exact_files = [f for f in listdir(exact_path) if isfile(join(exact_path, f))]

    for file in approx_files:

        split_string = file.split("_", 4)
        end = split_string[-1]
        typ = split_string[0]
        graphname = split_string[1]
        method = split_string[2]

        # Get all normalized files and the according exact values
        if typ == "Approx" and end == "True.txt":
            if graphname in graphs:
                if Original:
                    filename = f"{graphname}.txt"
                else:
                    filename = f"Exact_{graphname}_Abs_norm_True.txt"

                if f"{filename}" in exact_files:
                    exact = IO.file_to_dict(f"{exact_path}\\{filename}")
                    approx = IO.file_to_dict(f"{approx_path}\\{file}")

                    # Calculate absolute error
                    error = {
                        key: exact[key] - approx.get(key, 0) for key in exact.keys()
                    }

                    # Write Error Files
                    with open(
                        f"{error_path}\\Error_{graphname}_{method}.txt",
                        "w",
                    ) as fp:
                        fp.write(
                            "\n".join(
                                "{}:    {}".format(node, x) for node, x in error.items()
                            )
                        )
                    print(f"Generated {error_path}\\Error_{graphname}_{method}.txt")

                else:
                    raise Exception(f"{filename} seems to be missing")
    return


if __name__ == "__main__":
    sys.exit(main())
