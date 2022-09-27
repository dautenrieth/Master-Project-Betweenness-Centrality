"""
    In this script I experimented with the generation of graphs from scratch using configuration models
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import sys
import os

Project_Path = os.path.dirname(os.path.abspath(__file__))

number_of_runs = 10

n = 400
d = 3
"""
    Either vary the number of nodes or the degree to get the same connectivity
"""


def degreeconstant():
    """
    Generate graphs by using constant degree and varying size
    """
    for size in range(400, 4001, 40):
        ad_sum = 0  # Average Degree
        dia_sum = 0  # Diameter
        nn_sum = 0  # Number of Nodes in Giant Component
        dia = 0
        ad = 0
        nn = 0

        for run in range(1, number_of_runs + 1, 1):

            G = nx.configuration_model([d] * size)
            largest_cc = max(nx.connected_components(G), key=len)
            S = G.subgraph(largest_cc)
            # ad = sum([val for (node,val) in S.degree()])/S.number_of_nodes()
            # print(ad)
            dia = nx.diameter(S)
            nn = S.number_of_nodes()
            nx.write_edgelist(
                G,
                f"{Project_Path}\\Graphs Generated\\Configuration Model\\d constant\\CM_{size}_{dia}_{d}_{run}.edgelist",
                data=False,
            )
            print(f"CM_{size}_{dia}_{d}_{run}")
    print(f"run: {run}", end="\r")


def nconstant():
    """
    Generate graphs by using constant number of nodes and varying degree
    """
    for d in np.arange(2, 9, 1):

        ad_sum = 0  # Average Degree
        dia_sum = 0  # Diameter
        nn_sum = 0  # Number of Nodes in Giant Component
        dia = 0
        ad = 0
        nn = 0

        for run in range(1, number_of_runs + 1, 1):

            G = nx.configuration_model([d] * n)
            largest_cc = max(nx.connected_components(G), key=len)
            S = G.subgraph(largest_cc)
            ad = sum([val for (node, val) in S.degree()]) / S.number_of_nodes()
            print(ad)
            dia = nx.diameter(S)
            nn = S.number_of_nodes()
            nx.write_edgelist(
                G,
                f"{Project_Path}\\Graphs Generated\\Configuration Model\\n constant\CM_{n}_{dia}_{d}_{run}.edgelist",
                data=False,
            )
    print(f"run: {run}", end="\r")


def main():
    """
    Generating graphs
    """
    # Select method to use:
    NumberConstant = False
    DegreeConstant = True
    if NumberConstant:
        print("Started Generating Configuration Models with fixed Number of Nodes")
        nconstant()
    if DegreeConstant:
        print("Started Generating Configuration Models with fixed Degree")
        degreeconstant()


if __name__ == "__main__":
    sys.exit(main())
