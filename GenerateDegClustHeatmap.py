"""
    This module is used to generate different heatmaps
    TODO: - Can be intergrated in the plotting libary module
          - graph and methods as function attributes
"""
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import IO
from pathlib import Path
import os
import sys

graphs = ["ca-GrQc", "email-Enron", "ca-HepTh", "ca-HepPh", "com-amazon"]
methods = ["Abra", "DIAM", "RAND2"]

highest_number_cd = 40
highest_number_ed = 200
clustering_lower_bound = 0.05
clustering_upper_bound = 0.15


def CreateHeatmap(x, y, x_name, y_name, title, name):
    plt.hist2d(x, y, bins=50, cmap="plasma")
    cb = plt.colorbar()
    cb.set_label("Number of entries")
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(
        f"{os.path.dirname(os.path.abspath(__file__))}\\Plots\\Heatmaps\\{name}.png"
    )
    plt.clf()
    print(f"Created {name}.png")

    return


def ClusteringDegrees():
    for graph in graphs:
        # Load error file
        # error = IO.file_to_dict(f"{os.path.dirname(os.path.abspath(__file__))}\\Errors\\Error_{graph}_{method}.txt")
        G = nx.read_edgelist(
            f"{os.path.dirname(os.path.abspath(__file__))}\\Graphs/{graph}.lcc.net",
            nodetype=int,
        )
        C = nx.clustering(G)
        clusterings = [
            C[node] for node in list(G.nodes) if G.degree[node] < highest_number_cd
        ]
        degrees = [
            G.degree[node]
            for node in list(G.nodes)
            if G.degree[node] < highest_number_cd
        ]
        CreateHeatmap(
            clusterings,
            degrees,
            "Clustering Coefficients",
            "Degrees",
            "Heatmap Clustering vs Degrees",
            f"Clustering-Degree\\Clustering-Degree_{graph}",
        )


def ErrorClusteringDegree():
    for graph in graphs:
        for method in methods:
            # Load error file
            error = IO.file_to_dict(
                f"{os.path.dirname(os.path.abspath(__file__))}\\Errors\\Error_{graph}_{method}.txt"
            )
            G = nx.read_edgelist(
                f"{os.path.dirname(os.path.abspath(__file__))}\\Graphs/{graph}.lcc.net",
                nodetype=int,
            )
            C = nx.clustering(G)
            # clusterings = [C[node]  for node in list(G.nodes) if G.degree[node] < highest_number and C[node] > clustering_lower_bound and C[node] < clustering_upper_bound]
            degrees = [
                G.degree[node]
                for node in list(G.nodes)
                if G.degree[node] < highest_number_ed
                and C[node] > clustering_lower_bound
                and C[node] < clustering_upper_bound
            ]
            error_val = [
                error[node]
                for node in list(G.nodes)
                if G.degree[node] < highest_number_ed
                and C[node] > clustering_lower_bound
                and C[node] < clustering_upper_bound
            ]
            CreateHeatmap(
                error_val,
                degrees,
                "Error Rate",
                "Degrees",
                f"Heatmap Error vs Degrees, {clustering_lower_bound}<ClusteringCoeff<{clustering_upper_bound}",
                f"Error-Degree\\Error-Degree_{graph}_{method}",
            )


def ErrorClusteringDegreeWithError(show=False, bins=40):

    binned_degree = np.linspace(0, highest_number_cd, bins)
    binned_cc = np.linspace(0, 1, bins)
    for graph in graphs:
        for method in methods:
            clusterings = []
            degrees = []
            error_val = []
            # Load error file
            error = IO.file_to_dict(
                f"{os.path.dirname(os.path.abspath(__file__))}\\Errors\\Error_{graph}_{method}.txt"
            )
            exact_path = f"{os.path.dirname(os.path.abspath(__file__))}\\Exact_Betweenness\\Normalized_Scores"
            exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
            G = nx.read_edgelist(
                f"{os.path.dirname(os.path.abspath(__file__))}\\Graphs/{graph}.lcc.net",
                nodetype=int,
            )
            C = nx.clustering(G)
            for node in range(len(G.nodes)):
                if G.degree[node] < highest_number_cd:
                    clusterings.append(C[node])
                    degrees.append(G.degree[node])
                    if exact[node] != 0:
                        error_val.append(error[node] / exact[node])
                    else:
                        error_val.append(0.0)

            digitized_degree = np.digitize(degrees, binned_degree, right=True)
            digitized_cc = np.digitize(clusterings, binned_cc, right=True)

            average_relError = np.empty([len(binned_cc), len(binned_degree)])
            for x in range(len(binned_cc)):
                for y in range(len(binned_degree)):
                    if len(digitized_degree) != len(digitized_cc):
                        raise Exception("Something in the Binning Process didnt work")
                    else:
                        sum = 0
                        n = 0
                        for node in range(len(digitized_degree)):
                            if digitized_degree[node] == y and digitized_cc[node] == x:
                                sum += abs(error_val[node])
                                n += 1
                        if n != 0:
                            average_relError[y][x] = sum / n
                        else:
                            average_relError[y][x] = -1

            fig, ax = plt.subplots()
            im = ax.imshow(average_relError)

            # Show all ticks and label them with the respective list entries
            divide_parts = 10
            ax.set_xticks(
                np.arange(0, len(binned_cc), len(binned_cc) / divide_parts),
                labels=[round(x, 1) for x in np.arange(0, 1, 1 / divide_parts)],
            )
            ax.set_yticks(
                np.arange(0, len(binned_degree), len(binned_degree) / divide_parts),
                labels=[
                    round(x, 1)
                    for x in np.arange(
                        0, highest_number_cd, highest_number_cd / divide_parts
                    )
                ],
            )

            # Rotate the tick labels and set their alignment.
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Loop over data dimensions and create text annotations.
            # for i in range(len(binned_degree)):
            #     for j in range(len(binned_cc)):
            #         text = ax.text(j, i, average_relError[i, j],
            #                     ha="center", va="center", color="w")
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(
                "Average Relative Error (-1 -> no entities", rotation=-90, va="bottom"
            )
            ax.set_xlabel("Clustering Coefficent")
            ax.set_ylabel("Degree")
            ax.set_title(f"{graph} {method}")
            fig.tight_layout()
            if show:
                plt.show()
            fig.savefig(
                f"Plots\\Heatmaps\Clustering-Degree-RelError\\Clustering-Degree_w_RelError_{graph}_{method}.png"
            )
            print(
                f"Generated Plots\\Heatmaps\Clustering-Degree-RelError\\Clustering-Degree_w_RelError_{graph}_{method}.png"
            )
            plt.close()


def main():
    # Run Heatmap Generation for Clustering and Degree
    ClusteringDegrees()
    # More in depth analysis
    ErrorClusteringDegree()
    ErrorClusteringDegreeWithError()


if __name__ == "__main__":
    sys.exit(main())
