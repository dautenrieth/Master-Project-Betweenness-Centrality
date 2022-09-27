"""
With this module the different attributes of the error distribution are analyzed
Note: if approximation functions are run again, sample points have to be adjusted to generate the same results
"""

import IO
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os, sys
from sympy import symbols, Eq, solve


Project_Path = os.path.dirname(os.path.abspath(__file__))

graphs = ["ca-GrQc", "ca-HepTh", "ca-HepPh"]
# graphs = ["ca-GrQc"]
methods = ["Abra"]

# Type of function used for the deduction of the point distribution
def mfunction(x, a, b, c):
    """
    function for approximating the limits of a dotted line

    Args:
        x: variable
        a,b,c: factors

    Returns:
        Value
    """
    return a / -(x**b) + c  # must be the same as further down in solve function


def mfunctionY(y, a, b, c):
    """
    function for approximating the limits of a dotted line
    Solved for x

    Args:
        y: variable
        a,b,c: factors

    Returns:
        Value
    """
    if b == 0:
        return 10000
    elif a == 0:
        return 10000
    else:
        return ((c - y) / a) ** (-1 / b)  # solved upper function for x


def OneErrorNodesVisualisation():
    """
    Create visualization of node with error equal to one

    Args:
        None

    Returns:
        None
    """
    iNodes = []
    uNodes = []
    for index, graph in enumerate(graphs):
        for method in methods:
            print(f"Computing {graph} {method}")
            # Load error file
            error = IO.file_to_dict(
                f"{Project_Path}\\Errors\\Error_{graph}_{method}.txt"
            )
            G = nx.read_edgelist(
                f"{Project_Path}\\Graphs/{graph}.lcc.net", nodetype=int
            )
            exact_path = f"{Project_Path}\\Exact_Betweenness\\Normalized_Scores"
            exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
            all = {}
            for node in G.nodes:
                ex = exact[node]
                if ex != 0.0:
                    er = error[node]
                    result = er / ex
                    all[node] = result
                else:
                    all[node] = 0.0

            for node in G.nodes:
                if all[node] == 1:
                    iNodes.append(node)
                    ex = exact[node]
                    print(
                        f"Error Rate: {error[node]:.64f}, Exact Value: {exact[node]:.64f}, Relative Error: {all[node]}"
                    )
                    if exact[node] == 0.0:
                        print(f"This node: {node}")
                else:
                    uNodes.append(node)

            # for node in G.nodes:

            #     print(f'Error Rate: {error[node]}, Exact Value: {exact[node]}, Relative Error: {all_errorval[node]}')

            plt.figure(figsize=(50, 50))
            plt.title(f"{graph} Interesting Nodes")
            pos = nx.spring_layout(G)  # positions for all nodes

            nx.draw_networkx_nodes(G, pos, node_size=10, nodelist=uNodes)
            nx.draw_networkx_nodes(
                G, pos, node_size=30, nodelist=iNodes, node_color="r"
            )
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)

            plt.savefig(f"{Project_Path}\\Plots\\RErrorOneNodes_{graph}.png")
            print(f"Saved {Project_Path}\\Plots\\RErrorOneNodes_{graph}.png")
            # plt.show()
            plt.clf()
    return


def InterestingNodeSelection(
    NodeSelectionVisualization=False, InterestingNodeVisualization=False, show=False
):
    """
    Create visualizations for intersting nodes - nodes in one strand

    Args:
        NodeSelectionVisualization: Show selection lines visualization
        InterestingNodeVisualization: Show intersting node visualization

    """
    # Limits for Graphs
    # ["ca-GrQc", "email-Enron", "ca-HepTh", "ca-HepPh", "com-amazon"]
    upper_limit = [2, 2, 2]
    lower_limit = [-6, -7, -7]
    for index, graph in enumerate(graphs):
        for method in methods:
            print(f"Computing {graph} {method}")
            # Load error file
            error = IO.file_to_dict(
                f"{Project_Path}\\Errors\\Error_{graph}_{method}.txt"
            )
            G = nx.read_edgelist(
                f"{Project_Path}\\Graphs/{graph}.lcc.net", nodetype=int
            )
            exact_path = f"{Project_Path}\\Exact_Betweenness\\Normalized_Scores"
            exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
            C = nx.clustering(G)
            approx = IO.file_to_dict(
                f"{Project_Path}\\Approx_Betweenness\\Approx_{graph}_{method}_norm_True.txt"
            )

            # Sampled points for function approximation based on graphs
            if graph == "ca-GrQc":
                x = [
                    0.00013,
                    0.000033,
                    0.000013,
                    0.000009457,
                    0.00000775,
                    0.0000219,
                    0.000057,
                    0.0002,
                ]
                y = [0.71, 0.03, -1.58, -3.12, -4.28, -0.6, 0.37, 0.808]
                x2 = [
                    0.0001,
                    0.00005,
                    0.000028,
                    0.00001544,
                    0.00001,
                    0.000007,
                    0.0000047,
                ]
                y2 = [0.83, 0.67, 0.26, -0.36, -0.96, -2.2, -3.62]
                x3 = [
                    0.0002,
                    0.00015,
                    0.00008,
                    0.00005,
                    0.000036,
                    0.000023,
                    0.000017,
                    0.000014,
                ]
                y3 = [0.65, 0.566, 0.272, -0.266, -0.829, -1.86, -3.064, -4.526]
            elif graph == "ca-HepTh":
                x3 = [
                    0.00046,
                    0.00029,
                    0.00016,
                    0.0001,
                    0.0000697,
                    0.0000509,
                    0.000034,
                    0.0000266,
                    0.0000209,
                    0.0000135,
                    0.0000121,
                    0.00001,
                ]
                y3 = [
                    0.859,
                    0.738,
                    0.526,
                    0.257,
                    -0.162,
                    -0.539,
                    -1.317,
                    -2.018,
                    -2.767,
                    -4.493,
                    -5.479,
                    -6.954,
                ]
                x2 = [
                    0.0000033,
                    0.00000434,
                    0.00000578,
                    0.00000656,
                    0.00000927,
                    0.0000143,
                    0.000022,
                    0.0000288,
                    0.0000524,
                    0.0000928,
                    0.0002,
                    0.0004,
                ]
                y2 = [
                    -6.98,
                    -5.739,
                    -4.354,
                    -3.5,
                    -2.209,
                    -1.456,
                    -0.396,
                    -0.05,
                    0.413,
                    0.7,
                    0.885,
                    0.967,
                ]
                x = [
                    0.0004,
                    0.00024,
                    0.000088,
                    0.000048,
                    0.000033,
                    0.000022,
                    0.0000155,
                    0.0000116,
                    0.00000932,
                    0.00000655,
                    0.00000617,
                ]
                y = [
                    0.889,
                    0.798,
                    0.451,
                    -0.01,
                    -0.468,
                    -1.242,
                    -2.2,
                    -3.2,
                    -4.452,
                    -6.582,
                    -7.364,
                ]
            elif graph == "ca-HepPh":
                x = [
                    0.0007,
                    0.00039,
                    0.000174,
                    0.000091,
                    0.000056,
                    0.000038,
                    0.0000295,
                    0.0000275,
                    0.0000205,
                    0.000016,
                    0.0000149,
                    0.0000125,
                    0.0000109,
                ]
                y = [
                    0.866,
                    0.762,
                    0.45,
                    -0.062,
                    -0.718,
                    -1.5,
                    -2.293,
                    -2.504,
                    -3.785,
                    -4.785,
                    -5.56,
                    -6.99,
                    -8.1,
                ]
                x2 = [
                    0.00000517,
                    0.0000065,
                    0.0000078,
                    0.00000998,
                    0.0000142,
                    0.000023,
                    0.000037,
                    0.000073,
                    0.00016,
                    0.0003,
                    0.0007,
                ]
                y2 = [
                    -6.93,
                    -5.764,
                    -4.3,
                    -3.262,
                    -2.042,
                    -0.967,
                    -0.126,
                    0.504,
                    0.834,
                    0.924,
                    0.958,
                ]
                x3 = [
                    0.00069,
                    0.00035,
                    0.00016,
                    0.00011,
                    0.0000847,
                    0.0000576,
                    0.000043,
                    0.000034,
                    0.000032,
                    0.000024,
                    0.00002127,
                ]
                y3 = [
                    0.767,
                    0.53,
                    -0.027,
                    -0.472,
                    -0.944,
                    -1.944,
                    -2.8,
                    -3.87,
                    -4.62,
                    -5.54,
                    -7,
                ]
            else:
                raise Exception(f"{graph} doesnt seem to be implemented")

            if len(x) != 0:
                popt, _ = curve_fit(mfunction, x, y)
                popt2, _ = curve_fit(mfunction, x2, y2)
                popt3, _ = curve_fit(mfunction, x3, y3)
            # summarize the parameter values
            a, b, c = popt
            a2, b2, c2 = popt2
            a3, b3, c3 = popt3

            fpointsX = []
            fpointsY = []
            fpointsX2 = []
            fpointsY2 = []
            fpointsX3 = []
            fpointsY3 = []

            for xp in np.arange(0.000004, 0.0002, 0.000001):
                ty = mfunction(xp, a, b, c)
                ty2 = mfunction(xp, a2, b2, c2)
                ty3 = mfunction(xp, a3, b3, c3)
                # ty = func[0] * xp ** 2 + func[1] * xp **1 + func[2]
                if ty < 1 and ty > lower_limit[index]:
                    fpointsY.append(ty)
                    fpointsX.append(xp)
                if ty2 < 1 and ty2 > lower_limit[index]:
                    fpointsY2.append(ty2)
                    fpointsX2.append(xp)
                if ty3 < 1 and ty3 > lower_limit[index]:
                    fpointsY3.append(ty3)
                    fpointsX3.append(xp)

            # Plotting Clustering vs Error
            all_bcval = {}
            all_errorval = {}
            all_cval = {}
            for node in G.nodes:
                all_bcval[node] = exact[node]
                all_cval[node] = C[node]
                if exact[node] != 0:
                    all_errorval[node] = error[node] / exact[node]
                else:
                    all_errorval[node] = 0
                if error[node] != (exact[node] - approx[node]):
                    print("This doesnt work")

            iNodes = []
            iNodesX = []
            iNodesY = []
            uNodes = []
            iNodes2 = []
            iNodesX2 = []
            iNodesY2 = []

            # First set of interesting Nodes
            if graph == "ca-GrQc":
                upper_bound = 0.00018
                lower_bound = 0.000004
            elif graph == "ca-HepTh":
                upper_bound = 0.0006
                lower_bound = 0.000004
            elif graph == "ca-HepPh":
                upper_bound = 0.0006
                lower_bound = 0.000004
            else:
                raise Exception(f"{graph} doesnt seem to be implemented")

            for node in G.nodes:
                if (
                    all_bcval[node] > lower_bound and all_bcval[node] < upper_bound
                ):  # Limit area
                    func_val1 = mfunctionY(all_errorval[node], a, b, c)
                    func_val2 = mfunctionY(all_errorval[node], a2, b2, c2)
                    bc = all_bcval[node]
                    if (
                        bc < func_val1 and bc > func_val2
                    ):  # Check if node in between functions
                        iNodes.append(node)
                        iNodesX.append(all_bcval[node])
                        iNodesY.append(all_errorval[node])

            # Second set of interesting Nodes
            for node in G.nodes:
                if (
                    all_bcval[node] > lower_bound and all_bcval[node] < upper_bound
                ):  # Limit area
                    func_val1 = mfunctionY(all_errorval[node], a3, b3, c3)
                    func_val2 = mfunctionY(all_errorval[node], a, b, c)
                    bc = all_bcval[node]
                    if (
                        bc < func_val1 and bc > func_val2
                    ):  # Check if node in between functions
                        iNodes2.append(node)
                        iNodesX2.append(all_bcval[node])
                        iNodesY2.append(all_errorval[node])

            for node in G.nodes:
                if node not in iNodes and node not in iNodes2:
                    uNodes.append(node)

            if NodeSelectionVisualization:
                error_print = [
                    all_errorval[node] for node in range(len(all_errorval.keys()))
                ]
                bc_print = [all_bcval[node] for node in range(len(all_errorval.keys()))]
                c_print = [all_cval[node] for node in range(len(all_errorval.keys()))]
                plt.figure(figsize=(10, 10))
                plt.title(f"Relative Error - NBC {graph} {method}")
                plt.scatter(bc_print, error_print, c=c_print, cmap="viridis")
                plt.plot(fpointsX, fpointsY, "b")
                plt.plot(fpointsX2, fpointsY2, "g")
                plt.plot(fpointsX3, fpointsY3, "m")
                plt.plot(x, y, "bo")
                plt.plot(x2, y2, "go")
                plt.plot(x3, y3, "mo")
                plt.plot(iNodesX, iNodesY, "ro")
                plt.plot(iNodesX2, iNodesY2, "co")
                plt.xscale("log")
                plt.colorbar(label="Clustering Coefficent")
                plt.ylim([lower_limit[index], upper_limit[index]])
                plt.xlabel("Normalized Betweenness Centrality")
                plt.ylabel("Relative Error")
                plt.savefig(f"{Project_Path}\\Plots\\NodeSeperation_{graph}.png")
                print(f"{Project_Path}\\Plots\\NodeSeperation_{graph}.png")
                if show:
                    plt.show()
                plt.clf()

            # Color edges of adjacent interesting nodes
            colors = []
            for e in G.edges():
                if e[0] in iNodes and e[1] in iNodes:
                    colors.append("r")
                elif e[0] in iNodes2 and e[1] in iNodes2:
                    colors.append("c")
                else:
                    colors.append("k")

            if InterestingNodeVisualization:
                plt.figure(figsize=(50, 50))
                plt.title(f"{graph} Interesting Nodes")
                pos = nx.spring_layout(G)  # positions for all nodes

                nx.draw_networkx_nodes(
                    G, pos, node_size=10, nodelist=uNodes, node_color="k"
                )
                nx.draw_networkx_nodes(
                    G, pos, node_size=30, nodelist=iNodes, node_color="r"
                )
                nx.draw_networkx_nodes(
                    G, pos, node_size=30, nodelist=iNodes2, node_color="c"
                )
                nx.draw_networkx_edges(G, pos, edge_color=colors, alpha=0.5, width=1)

                plt.savefig(f"{Project_Path}\\Plots\\InterestingNodes_{graph}.png")
                print(f"Saved {Project_Path}\\Plots\\InterestingNodes_{graph}.png")
                if show:
                    plt.show()
                plt.clf()

            # Check if Nodes are neighbors if output = 2 then they are

            NodeGroups = [iNodes, iNodes2]
            for index, NodeGroup in enumerate(NodeGroups):
                print(f"Group {index+1} of interesting Nodes:")
                spg = []
                sample_value = set()
                number_of_zero_values = 0
                for node in NodeGroup:
                    sp = []
                    for node2 in NodeGroup:
                        if node != node2:
                            sp.append(
                                len(nx.shortest_path(G, source=node, target=node2))
                            )
                    spg.append(min(sp))
                    if approx[node] == 0:
                        number_of_zero_values += 1
                        if exact[node] == 0:
                            print(f"Oh no! {all_errorval[node]}")
                    else:
                        sample_value.add(
                            round(
                                (index + 1)
                                / (exact[node] - (all_errorval[node] * exact[node]))
                            )
                        )
                    # print(f'approx {node}: {approx[node]}')
                print(
                    f"From {len(NodeGroup)} interesting nodes, {number_of_zero_values} nodes have a value of 0"
                )
                if len(sample_value) > 1:
                    print(
                        f"Please check Group {index+1} selection again. it looks like points have been selected outside the curve: Sample-Values: {sample_value}"
                    )
                else:
                    print(f"{sample_value} paths were sampled")
                print(f"Shortest Paths: {spg}\n")
    return


def main():
    """
    Run all availabe functions
    """
    OneErrorNodesVisualisation()
    InterestingNodeSelection(True, False, show=False)


if __name__ == "__main__":
    sys.exit(main())
