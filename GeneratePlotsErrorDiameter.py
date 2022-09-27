"""
This module allows the visualization of error values against the diameter for generated graphs
Can be merged with the Plotting Libary later on. But considering the different naming conventions
"""

# Imports
import networkx as nx
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import IO
from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import sys

avd = 4
n = 400
d = 10
N = 2000

methods = ["Abra", "DIAM", "RAND2"]
Project_Path = os.path.dirname(os.path.abspath(__file__))


def calculateAverageError(file: str, error_path: Path) -> float:
    """
    Simple method to calculate the average error

    Args:
        file: name of the file
        error_path: path to error file

    Returns:
        average error
    """
    error = IO.file_to_dict(f"{error_path}\\{file}")
    return sum([abs(val) for val in error.values()]) / len(error)


def get_values(error_path: Path, output_variable: str):
    """
    Fetch error data and calculate average errors. Give back different parameters

    Args:
        error_path: path to error file
        output_variable: type of varied variable in generation method (e.g. d, N, n)

    Returns:
        The diameter, average error and property of the fetched file
    """
    error_files = [f for f in listdir(error_path) if isfile(join(error_path, f))]

    ER_diameter = [[] for _ in range(len(methods))]
    ER_error = [[] for _ in range(len(methods))]

    for file in error_files:
        split_string = file.split("_", 6)
        # Only select files with specific properties
        if (
            (int(split_string[4]) == avd and output_variable == "avd")
            or (int(split_string[2]) == n and output_variable == "n")
            or (output_variable == "d")
            or (int(split_string[2]) == N and output_variable == "N")
        ):

            if output_variable == "N":  # Necessary because of new format with runs
                ER_diameter[methods.index(split_string[5].split(".", 1)[0])].append(
                    int(split_string[3])
                )
                ER_error[methods.index(split_string[5].split(".", 1)[0])].append(
                    calculateAverageError(file, error_path)
                )
            else:
                ER_diameter[methods.index(split_string[6].split(".", 1)[0])].append(
                    int(split_string[3])
                )
                ER_error[methods.index(split_string[6].split(".", 1)[0])].append(
                    calculateAverageError(file, error_path)
                )
            if output_variable == "avd":
                prop = split_string[4]
            elif output_variable == "n":
                prop = split_string[2]
            elif output_variable == "N":
                prop = split_string[2]
            else:
                prop = split_string[4]

    if len(ER_diameter[0]) == 0:
        raise Exception(
            "No files with matching properties found. Please adjust search parameters or a files"
        )

    return ER_diameter, ER_error, prop


def run(output_variable, error_path, typ, title, log=False):
    """
    Function to visualize diameter and error data

    Args:
        output_variable: type of varied variable in generation method (e.g. d, N, n)
        error_path: path to error file
        typ: type e.g. ER (Erdos Renyi) or CM (configuration model). Used for filename
        title: title of the plot
        log: Should the plot have a logarithmic x-axis?

    Returns:
        Nothing - All outputs can be found in the Plots folder

    """
    ER_diameter, ER_error, c = get_values(error_path, output_variable)
    Error = [[] for _ in range(len(methods))]
    Average_Error = [[] for _ in range(len(methods))]
    Diameter = [[] for _ in range(len(methods))]
    Counter = [[] for _ in range(len(methods))]
    for m in range(len(methods)):
        for i, elem in enumerate(ER_diameter[m]):
            if elem not in Diameter[m]:
                Diameter[m].append(elem)
                index = Diameter[m].index(elem)
                Error[m].append([ER_error[m][i]])
                Counter[m].append(1)
            else:
                index = Diameter[m].index(elem)
                Error[m][index].append(ER_error[m][i])
                Counter[m][index] += 1
    for m in range(len(methods)):
        for i, count in enumerate(Counter[m]):
            Average_Error[m].append(np.sum(Error[m][i]) / count)

    # Sort Average Errors
    Diameter[0], Average_Error[0] = [
        list(v) for v in zip(*sorted(zip(Diameter[0], Average_Error[0])))
    ]
    Diameter[1], Average_Error[1] = [
        list(v) for v in zip(*sorted(zip(Diameter[1], Average_Error[1])))
    ]
    Diameter[2], Average_Error[2] = [
        list(v) for v in zip(*sorted(zip(Diameter[2], Average_Error[2])))
    ]

    plt.title(f"{title}{c}")
    plt.plot(ER_diameter[0], ER_error[0], "bo", label=f"{methods[0]}")
    plt.plot(ER_diameter[1], ER_error[1], "ro", label=f"{methods[1]}")
    plt.plot(ER_diameter[2], ER_error[2], "go", label=f"{methods[2]}")
    plt.plot(Diameter[0], Average_Error[0], color="b")
    plt.plot(Diameter[1], Average_Error[1], color="r")
    plt.plot(Diameter[2], Average_Error[2], color="g")
    if log:
        plt.xscale("log")
    plt.legend()
    plt.xlabel("Diameter")
    plt.ylabel("Average Error")
    plt.savefig(
        f"{os.path.dirname(os.path.abspath(__file__))}\\Plots\\Diameter\\Diameter-Error_{typ}_{output_variable}{c}.png"
    )
    plt.clf()
    print(
        f"Saved {os.path.dirname(os.path.abspath(__file__))}\\Plots\\Diameter\\Diameter-Error_{typ}_{output_variable}{c}.png"
    )
    return


def main(
    Configurationmodel_nconstant=True,
    Configurationmodel_dconstant=True,
    ErdosRenyi=False,
    ErdosRenyiN=True,
):
    """
    Selection of the different generation options.
    Make sure graphs and accoring files are generated before activting options

    Args:
        Configurationmodel_nconstant: a graph generated as a configuration model with a constant number of nodes
        Configurationmodel_dconstant: a graph generated as a configuration model with a constant degree of a node
        ErdosRenyi: An Erdos Renyi graph with a constant c
        ErdosRenyiN: An Big Erdos Renyi graph with a big number of nodes N

    Returns:
        Nothing - All files and visualizations will be saved locally
    """

    # Select for which type of graphs you want to run the process
    if Configurationmodel_nconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant n"
        )
        run(
            "n",
            Path(
                f"{Project_Path}\\Graphs Generated\\Configuration Model\\n constant\\Errors"
            ),
            "CM",
            title="Configuration Model Number of Nodes=",
            log=True,
        )

    if Configurationmodel_dconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant degree"
        )
        run(
            "avd",
            Path(
                f"{Project_Path}\\Graphs Generated\\Configuration Model\\d constant\\Errors"
            ),
            "CM",
            title="Configuration Model Degree=",
        )

    if ErdosRenyi:
        print("Started to run Approximations for Erdos Renyi Graphs")
        run(
            "d",
            Path(f"{Project_Path}\\Graphs Generated\\Erdos Renyi\\Errors"),
            "ER",
            title="Erdos renyi Graph c=",
        )

    if ErdosRenyiN:
        print("Started to run Approximations for Erdos Renyi Graphs")
        run(
            "N",
            Path(f"{Project_Path}\\Graphs Generated\\Erdos Renyi Big\\Errors"),
            "ERBIG",
            title="Erdos renyi Graph N=",
        )
    return


if __name__ == "__main__":
    sys.exit(main())
