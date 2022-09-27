"""
    This module contains all important functions to produce different visualizations based on the existing data.

    Typical usage example:
        PlottingLibary.functionx([List of graphs]) -> generates plots
"""

# Imports
import IO
import PlottingHelperLibary as ph
import HelperFunctions

import networkx as nx
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import sys
import os

Project_Path = os.path.dirname(os.path.abspath(__file__))


def plot_error_clustering(
    graphs: List[str],
    methods: List[str],
    relative=False,
    show=False,
    path_add_error: str = "",
    path_add_exact: str = "",
    Original: bool = True,
):
    """
    Plots the Clustering coefficient against the Degree.
    It is possible to choose between the originals and the generated graphs (Original = True/False).

    Args:
        graphs: list of graphs for which a plot is to be created
        methods: the methods to be used
        relative: whether the betweenness centrality should be displayed absolutely or realtively
        show: Whether or not the generated plots should be displayed during execution
        path_add_error: The subfolder of the error data
        path_add_exact: the subfolder of the exact betweenness centrality data
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """

    upperlimit = 0.003
    lowerlimit = -0.003
    upperlimitRel = 5
    lowerlimitRel = -5

    # List of implemented methods. If you add a methods, please also add naming here
    HelperFunctions.MethodComparison(["Abra", "DIAM", "RAND2"], methods)

    for graph in graphs:
        for method in methods:
            # Load error file
            error = IO.file_to_dict(
                f"{Project_Path}\\{path_add_error}\\Errors\\Error_{graph}_{method}.txt"
            )
            G = nx.read_edgelist(
                f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int
            )
            C = nx.clustering(G)
            # Plotting Clustering vs Error
            if relative:
                if Original:
                    exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness\\Normalized_Scores"
                else:
                    exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness"
                    exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")

            all_clusterval = []
            all_errorval = []
            for node in G.nodes:
                all_clusterval.append(C[node])
                if relative:
                    all_errorval.append(
                        error[node] / exact[node] if exact[node] != 0 else 0
                    )
                else:
                    all_errorval.append(error[node])

            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.005

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + spacing, width, 0.2]

            fig = plt.figure(figsize=(10, 12))
            ax = fig.add_axes(rect_scatter)
            fig.suptitle(
                f"{graph}  {method}",
                fontsize=16,
                x=0.35,
                y=0.99,
                horizontalalignment="left",
            )
            ax.set_xlabel("Clustering Coefficent")
            ax_histx = fig.add_axes(rect_histx, sharex=ax)

            # use the previously defined function
            bins = ph.binning(all_clusterval)
            ph.scatter_hist(all_clusterval, all_errorval, ax, ax_histx, bins)
            y_av = ph.movingaverage(all_clusterval, all_errorval, bins)

            if relative:
                ax.set_ylabel("Relative Error")
                ax.plot(bins, y_av, "r", label="Relative Average Error")
                ax.legend()
                ax.set_ylim([lowerlimitRel, upperlimitRel])
                plt.savefig(
                    f"Plots\\Clustering\\Relative Error\\Clustering-Rel_Error_{graph}_{method}.png"
                )
                print(
                    f"Generated Plots\\Clustering\\Relative Error\\Clustering-Rel_Error_{graph}_{method}.png"
                )
            else:
                ax.set_ylabel("Absolute Error")
                ax.plot(bins, y_av, "r", label="Absolute Average Error")
                ax.legend()
                ax.set_ylim([lowerlimit, upperlimit])
                plt.savefig(
                    f"Plots\\Clustering\\Absolute Error\\Clustering-Error_{graph}_{method}.png"
                )
                print(
                    f"Generated Plots\\Clustering\\Absolute Error\\Clustering-Error_{graph}_{method}.png"
                )

            if show:
                plt.ion()
                plt.show(block=True)
            else:
                plt.close()
            plt.clf()
            plt.close(fig)


def plot_error_degree(
    graphs: List[str],
    methods: List[str],
    relative: bool = False,
    log: bool = True,
    show: bool = False,
    path_add="",
    output_dir_add="",
    Original: bool = True,
):
    """
    Plots the Error against the Degree.
    It is possible to choose between the originals and the generated graphs (Original = True/False).

    Args:
        graphs: list of graphs for which a plot is to be created
        methods: the methods to be used
        relative: whether the betweenness centrality should be displayed absolutely or realtively
        log: Should the plot have a logarithmic x-axis?
        show: Whether or not the generated plots should be displayed during execution
        path_add: The folder of generated graphs. This is also used in the output. For more details see folder structure
        output_dir_add: subfolders for the outputs (e.g. for generated graphs)
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """
    upperlimit = 0.003
    lowerlimit = -0.003
    upperlimitRel = 5
    lowerlimitRel = -5
    # List of implemented methods. If you add a methods, please also add naming here
    HelperFunctions.MethodComparison(["Abra", "DIAM", "RAND2"], methods)
    for graph in graphs:
        for method in methods:
            # Load error file
            error = IO.file_to_dict(
                f"{Project_Path}\\{path_add}\\Errors\\Error_{graph}_{method}.txt"
            )
            if Original:
                G = nx.read_edgelist(
                    f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int
                )
            else:
                G = nx.read_edgelist(
                    f"{Project_Path}\\{path_add}\\{graph}.edgelist", nodetype=int
                )
            if relative:
                if Original:
                    exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness\\Normalized_Scores"
                    exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
                else:
                    exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness"
                    exact = IO.file_to_dict(
                        f"{exact_path}\\Exact_{graph}_Abs_norm_True.txt"
                    )

            all_degreeval = []
            all_errorval = []
            for node in G.nodes:
                all_degreeval.append(G.degree[node])
                if relative:
                    all_errorval.append(
                        error[node] / exact[node] if exact[node] != 0 else 0
                    )
                else:
                    all_errorval.append(error[node])

            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.005

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + spacing, width, 0.2]

            fig = plt.figure(figsize=(10, 12))
            ax = fig.add_axes(rect_scatter)
            fig.suptitle(
                f"{graph}  {method}",
                fontsize=16,
                x=0.35,
                y=0.99,
                horizontalalignment="left",
            )
            ax.set_xlabel("Degree")

            ax_histx = fig.add_axes(rect_histx, sharex=ax)

            # use the previously defined function
            bins = ph.binning(all_degreeval, linear=False)
            ph.scatter_hist(all_degreeval, all_errorval, ax, ax_histx, bins)
            y_av = ph.movingaverage(all_degreeval, all_errorval, bins)
            # plt.plot(all_clusterval, all_errorval)
            ax.plot(bins, y_av, "r", label="Absolute Average Error")
            if relative:
                ax.set_ylim([lowerlimitRel, upperlimitRel])
                ax.set_ylabel("Relative Error")
            else:
                ax.set_ylim([lowerlimit, upperlimit])
                ax.set_ylabel("Absolute Error")
            ax.legend()
            if log:
                ax.set_xscale("log")
                if relative:
                    savename = f"Plots\\{output_dir_add}\\Degree\\Relative Error\\Logarithmic\\Degree-Error_{graph}_{method}_relative_log.png"
                else:
                    savename = f"Plots\\{output_dir_add}\\Degree\\Absolute Error\\Logarithmic\\Degree-Error_{graph}_{method}_log.png"
            else:
                if relative:
                    savename = f"Plots\\{output_dir_add}\\Degree\\Relative Error\\Linear\\Degree-Error_{graph}_{method}_relative_log.png"
                else:
                    savename = f"Plots\\{output_dir_add}\\Degree\\Absolute Error\\Linear\\Degree-Error_{graph}_{method}.png"
            plt.savefig(savename)
            print(f"Generated {savename}")
            if show:
                plt.ion()
                plt.show(block=True)
            else:
                plt.ioff()
            plt.clf()
            plt.close(fig)


def plot_error_bc(
    graphs: List[str],
    methods: List[str],
    relative: bool = True,
    path_add="",
    path_add_exact="",
    output_dir_add="",
    show: bool = False,
    log: bool = False,
    Original: bool = True,
    Regression: bool = False,
    Regression_model: str = "RM",
):
    """
    Plots the Betweenness Centrality against the Degree.
    It is possible to choose between the originals and the generated graphs (Original = True/False).

    Args:
        graphs: list of graphs for which a plot is to be created
        methods: the methods to be used
        relative: whether the betweenness centrality should be displayed absolutely or realtively
        path_add: The folder of generated graphs. This is also used in the output. For more details see folder structure
        path_add_exact: If the exact betweenness centrality values are not in the default Exact_Betweenness folder but in subfolders (for example of generated graphs) the additional folder(s) must be passed here
        output_dir_add: subfolders for the outputs (e.g. for generated graphs)
        show: Whether or not the generated plots should be displayed during execution
        log: Should the plot have a logarithmic x-axis?
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?
        Regression: should the error data of regression models be used
        Regression_model: name of the regression model

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """
    for graph in graphs:
        for method in methods:
            # Load error file
            if Regression:
                error = IO.file_to_dict(
                    f"{Project_Path}\\{path_add}\\Errors\\Error_{graph}_{method}_{Regression_model}.txt"
                )
            else:
                error = IO.file_to_dict(
                    f"{Project_Path}\\{path_add}\\Errors\\Error_{graph}_{method}.txt"
                )

            if Original:
                G = nx.read_edgelist(
                    f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int
                )
            else:
                G = nx.read_edgelist(
                    f"{Project_Path}\\{path_add}\\{graph}.edgelist", nodetype=int
                )

            if Original:
                exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness\\Normalized_Scores"
                exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
            else:
                exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness"
                exact = IO.file_to_dict(
                    f"{exact_path}\\Exact_{graph}_Abs_norm_True.txt"
                )

            all_bcval = []
            all_errorval = []
            for node in G.nodes:
                all_bcval.append(exact[node])
                if relative:
                    all_errorval.append(
                        error[node] / exact[node] if exact[node] != 0 else 0
                    )
                    # if exact[node] != 0:
                    #     if error[node] / exact[node] < -4:
                    #         print(f"Not in Plot: {error[node] / exact[node]}")
                else:
                    all_errorval.append(error[node])

            plt.title(f"Error - NBC {graph} {method}")
            plt.plot(all_bcval, all_errorval, "bo")

            plt.xlabel("Normalized Betweenness Centrality")
            plt.ylabel("Error")
            if relative:
                plt.ylim([-10, 1.5])
                output_path = f"{Project_Path}\\Plots\\{output_dir_add}\\Betweenness Centrality\\Relative Error"
            else:
                plt.ylim([-0.004, 0.004])
                output_path = f"{Project_Path}\\Plots\\{output_dir_add}\\Betweenness Centrality\\Absolute Error"
            if log:
                plt.xscale("log")
                plt.savefig(f"{output_path}\\BC-Error_{graph}_{method}_log.png")
                print(f"Generated {output_path}\\BC-Error_{graph}_{method}_log.png")
            else:
                plt.savefig(f"{output_path}\\BC-Error_{graph}_{method}.png")
                print(f"Generated {output_path}\\BC-Error_{graph}_{method}.png")

            if show:
                plt.ion()
                plt.show(block=True)
            else:
                plt.ioff()
            plt.clf()
    return


def plot_degree_bc(
    graphs: List[str],
    path_add="",
    path_add_exact="",
    output_dir_add="",
    show: bool = False,
    Original: bool = True,
    withClustering: bool = True,
):
    """
    Plots the Betweenness Centrality against the Degree.
    It is possible to choose between the originals and the generated graphs (Original = True/False).

    Args:
        graphs: list of graphs for which a plot is to be created
        path_add: The folder of generated graphs. This is also used in the output. For more details see folder structure
        path_add_exact: If the exact betweenness centrality values are not in the default Exact_Betweenness folder but in subfolders (for example of generated graphs) the additional folder(s) must be passed here
        show: Whether or not the generated plots should be displayed during execution
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?
        withClustering: gives the plot a third (colored) dimension using the clustering coefficient

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """
    for graph in graphs:

        if Original:
            G = nx.read_edgelist(
                f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int
            )
        else:
            G = nx.read_edgelist(
                f"{Project_Path}\\{path_add}\\{graph}.edgelist", nodetype=int
            )

        if Original:
            exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness\\Normalized_Scores"
            exact = IO.file_to_dict(f"{exact_path}\\{graph}.txt")
        else:
            exact_path = f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness"
            exact = IO.file_to_dict(f"{exact_path}\\Exact_{graph}_Abs_norm_True.txt")

        if withClustering:
            C = nx.clustering(G)

        all_bcval = []
        all_degreeval = []
        all_cval = []
        for node in G.nodes:
            all_bcval.append(exact[node])
            all_degreeval.append(G.degree[node])
            if withClustering:
                all_cval.append(C[node])

        plt.title(f"Degree - NBC {graph}")
        plt.xlabel("Normalized Betweenness Centrality")
        plt.ylabel("Degree")

        output_path = f"{Project_Path}\\Plots\\{output_dir_add}\\Degree\\BC"

        if withClustering:
            plt.scatter(all_bcval, all_degreeval, c=all_cval, cmap="viridis")
            plt.colorbar(label="Clustering Coefficent")
            plt.savefig(f"{output_path}\\BC-Degree_{graph}_wC.png")
            print(f"Generated {output_path}\\BC-Degree_{graph}_wC.png")
        else:
            plt.plot(all_bcval, all_degreeval, "bo")
            plt.savefig(f"{output_path}\\BC-Degree_{graph}.png")
            print(f"Generated {output_path}\\BC-Degree_{graph}.png")

        if show:
            plt.ion()
            plt.show(block=True)
        else:
            plt.ioff()
        plt.clf()
    return


def plot_clustering_degree(
    graphs: List[str], path_add="", show: bool = False, Original: bool = True
):
    """
    Plots the Clustering Coefficent against the Degree.
    It is possible to choose between the originals and the generated graphs (Original = True/False).

    Args:
        graphs: list of graphs for which a plot is to be created
        path_add: The folder of generated graphs. This is also used in the output. For more details see folder structure
        show: Whether or not the generated plots should be displayed during execution
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """
    for graph in graphs:

        if Original:
            G = nx.read_edgelist(
                f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int
            )
        else:
            G = nx.read_edgelist(
                f"{Project_Path}\\{path_add}\\{graph}.edgelist", nodetype=int
            )
        output_path = f"{Project_Path}\\Plots\\{path_add}\\Clustering\\Degree"

        C = nx.clustering(G)
        all_degreeval = []
        all_clusteringval = []
        for node in G.nodes:
            all_degreeval.append(G.degree[node])
            all_clusteringval.append(C[node])
        plt.plot(all_clusteringval, all_degreeval, "o")
        plt.xlabel("Clustering Coefficent")
        plt.ylabel("Degree")
        plt.title(f"Clustering Degree {graph}")
        plt.savefig(f"{output_path}\\Clustering-Degree_{graph}.png")
        print(f"Generated {output_path}\Clustering-Degree_{graph}.png")
        if show:
            plt.ion()
            plt.show(block=True)
        else:
            plt.ioff()
        plt.clf()
    return


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
    methods: List[str] = ["Abra", "DIAM", "RAND2"],
):
    """
    Runs all implemented options of the Plotting Libary with standard attributes.
    """
    plot_error_clustering(graphs, methods, relative=False, show=False)
    plot_error_clustering(graphs, methods, relative=True, show=False)

    plot_error_degree(graphs, methods, relative=False, log=True)
    plot_error_degree(graphs, methods, relative=False, log=False)
    plot_error_degree(graphs, methods, relative=True, log=True)
    plot_error_degree(graphs, methods, relative=True, log=False)

    plot_error_bc(graphs, methods, relative=False, log=True)
    plot_error_bc(graphs, methods, relative=False, log=False)
    plot_error_bc(graphs, methods, relative=True, log=True)
    plot_error_bc(graphs, methods, relative=True, log=False)

    plot_degree_bc(graphs=graphs)
    plot_clustering_degree(graphs=graphs)


if __name__ == "__main__":
    sys.exit(
        main(["ca-GrQc", "email-Enron", "ca-HepTh", "ca-HepPh", "com-amazon"])
    )  # graph list without com-lj and dbpedia-link
