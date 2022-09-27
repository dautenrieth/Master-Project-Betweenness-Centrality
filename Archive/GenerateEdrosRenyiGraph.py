"""
    Simple script to generate different Erdos Renyi Graphs
"""

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import json
import RunApproximations as ra


number_of_runs = 1
c = 8


for p in np.arange(0.01, 0.2, 0.001):
    n = round(c / p)
    if np.log(n) / n >= p:
        print(f"p not large enough: {p}")
    ad_sum = 0  # Average Degree
    dia_sum = 0  # Diameter
    nn_sum = 0  # Number of Nodes in Giant Component
    dia = 0
    ad = 0
    nn = 0
    error_diam, error_rand2, error_abra = 0, 0, 0
    for run in range(1, number_of_runs + 1, 1):

        G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
        largest_cc = max(nx.connected_components(G), key=len)
        S = G.subgraph(largest_cc)
        ad = sum([val for (node, val) in S.degree()]) / S.number_of_nodes()
        print(ad)
        dia = nx.diameter(S)
        nn = S.number_of_nodes()
        nx.write_edgelist(
            G,
            f"C:\\Users\\Daniel\\Desktop\\Master Project\\Graphs Generated\\ER_{n}_{dia}_{c}_{run}.edgelist",
            data=False,
        )
print(f"run: {run}", end="\r")
