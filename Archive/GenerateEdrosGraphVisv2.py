"""
    Not actively used
    Script used to generate Erdos Renyi Graphs. Visualize them and save  parameters like:
        Probability, Diameter, AverageDegree, NumberNodesGiant
    Another way of generation compared to V1
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import json


def Visualization():
    diameters = []
    probilities = []
    for key in val_dict["NumberNodes"].keys():
        for prob in val_dict["NumberNodes"][key]["Probability"]:
            diameters.append(
                val_dict["NumberNodes"][key]["Probability"][prob]["Diameter"]
            )
            probilities.append(prob)
    # Visualisation
    plt.title(f"Erdos Renyi Graph with c={c}")
    plt.plot(probilities, diameters)
    plt.xlabel("Probability")
    plt.ylabel("Average Diameter")
    plt.savefig(f"Diameter-Probability-ER_c{c}.png")
    plt.clf()
    print(f"Saved Diameter-Probability-ERc{c}.png")
    return


val_dict = {"NumberNodes": {}}
number_of_runs = 5
c = 8


for p in np.arange(0.01, 0.045, 0.001):
    n = round(c / p)
    if np.log(n) / n >= p:
        print(f"p not large enough: {p}")
    ad_sum = 0  # Average Degree
    dia_sum = 0  # Diameter
    nn_sum = 0  # Number of Nodes in Giant Component
    dia = 0
    ad = 0
    nn = 0
    for run in range(1, number_of_runs + 1, 1):

        G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
        largest_cc = max(nx.connected_components(G), key=len)
        S = G.subgraph(largest_cc)
        ad_sum += sum([val for (node, val) in S.degree()]) / S.number_of_nodes()
        dia_sum += nx.diameter(S)
        nn_sum += S.number_of_nodes()
    ad = ad_sum / number_of_runs
    dia = dia_sum / number_of_runs
    nn = nn_sum / number_of_runs

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
Visualization()


with open(f"ER-Diameter-data-c{c}.json", "w") as fp:
    json.dump(val_dict, fp)
