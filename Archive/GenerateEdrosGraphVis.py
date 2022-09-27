"""
    Not actively used
    Script used to generate Erdos Renyi Graphs. Visualize them and save  parameters like:
        Probability, Diameter, AverageDegree, NumberNodesGiant
    Another way of generation compared to V2
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import json


def Visualization(NN):
    diameters = []
    probilities = []
    for prob in val_dict["NumberNodes"][NN]["Probability"]:
        diameters.append(val_dict["NumberNodes"][NN]["Probability"][prob]["Diameter"])
        probilities.append(prob)
    # Visualisation
    plt.title(f"Erdos Renyi Graph {NN} Nodes")
    plt.plot(probilities, diameters)
    plt.xlabel("Probability")
    plt.ylabel("Average Diameter")
    plt.savefig(f"Diameter-Probability-ER{NN}.png")
    plt.clf()
    print(f"Saved Diameter-Probability-ER{NN}.png")
    return


val_dict = {"NumberNodes": {}}
number_of_runs = 10


for n in range(1000, 2001, 1000):
    for p in np.arange(1 / 10000, 1 / 50, 1 / 10000):
        ad = 0  # Average Degree
        dia = 0  # Diameter
        nn = 0  # Number of Nodes in Giant Component
        for run in range(1, number_of_runs + 1, 1):

            G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
            largest_cc = max(nx.connected_components(G), key=len)
            S = G.subgraph(largest_cc)
            ad = sum([val for (node, val) in S.degree()]) / S.number_of_nodes()
            dia = nx.diameter(S)
            print(dia)
            nn = S.number_of_nodes()
        ad = ad / number_of_runs
        dia = dia / number_of_runs
        nn = nn / number_of_runs

        # Generate and fill Dictionary
        if n not in val_dict["NumberNodes"]:
            val_dict["NumberNodes"][n] = {}
        if "Probability" not in val_dict["NumberNodes"][n]:
            val_dict["NumberNodes"][n]["Probability"] = {}
        val_dict["NumberNodes"][n]["Probability"][p] = {
            "Diameter": dia,
            "AverageDegree": ad,
            "NumberNodesGiant": nn,
        }
        print(f"n: {n}, p: {p}")
    Visualization(n)


with open("ER-Diameter-data.json", "w") as fp:
    json.dump(val_dict, fp)
