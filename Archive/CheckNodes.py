"""
    Deprecated module. Better functionality integrated in the GetNodeBranches module 
"""

import os
import IO
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network

Project_Path = os.path.dirname(os.path.abspath(__file__))
print(Project_Path)
graph = "ca-GrQc"

error = IO.file_to_dict(f"{Project_Path}\\Errors\\Error_{graph}_DIAM.txt")
G = nx.read_edgelist(f"{Project_Path}\\Graphs/{graph}.lcc.net", nodetype=int)
exact_path = f"{Project_Path}\\Exact_Betweenness\\Normalized_Scores"
exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")

iNodes = []  # Interesting nodes
uNodes = []  # Normal Nodes
for node in G.nodes:
    if exact[node] != 0:
        if error[node] / exact[node] < -4.5:
            iNodes.append(node)
        else:
            uNodes.append(node)
    else:
        uNodes.append(node)
print(f"Interesting Nodes: {iNodes}")

for node in iNodes:
    sp = []
    for node2 in iNodes:
        if node != node2:
            sp.append(len(nx.shortest_path(G, source=node, target=node2)))
    print(min(sp))


plt.figure(figsize=(50, 50))
pos = nx.spring_layout(G)  # positions for all nodes

nx.draw_networkx_nodes(G, pos, node_size=4, nodelist=uNodes)
nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=iNodes, node_color="r")
nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)


plt.savefig(f"{Project_Path}\\Plots\\InterestingNodes_{graph}.png")
print(f"Saved {Project_Path}\\Plots\\InterestingNodes_{graph}.png")
plt.show()
plt.clf()
