"""
Simple script for adding up the total erros of the given graphs
"""

import os
import sys
import IO

# Change graphs here
graphs = ["ca-GrQc", "ca-HepPh", "ca-HepTh"]

Project_Path = os.path.dirname(os.path.abspath(__file__))


def totalError(methods, path_add="", CM: bool = False, Regression: bool = False):
    for graph in graphs:
        if CM:
            graphname = f"ConfigurationModel-{graph}-Distribution"
        else:
            graphname = graph
        for method in methods:
            total = 0
            error = IO.file_to_dict(
                f"{Project_Path}\\{path_add}\\Errors\\Error_{graphname}_{method}_RMI.txt"
            )
            for node, e in error.items():
                total += abs(e)
            print(f"{graphname}-{method}: {total}")
    return


def main():
    # totalError(methods = ["Abra", "DIAM", "RAND2"])
    # totalError(methods = ["Abra", "DIAM", "RAND2"], "Graphs Generated\Configuration Models Graph Distribution", CM=True)
    totalError(methods=["Abra"], path_add="Regression", Regression=True)
    return


if __name__ == "__main__":
    sys.exit(main())
