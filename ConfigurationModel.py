"""
    This module contains functions for creating configuration models, generating data with these models and visualizing the data.
    In the main function these functions are linked with a logic to give the user different application options.
"""

import sys, os
from typing import List, Optional, Tuple, Dict
import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import numpy as np
import RunApproximations as approx
import HelperFunctions
import IO
import GenerateErrorFiles
import PlottingLibary


Project_Path = os.path.dirname(os.path.abspath(__file__))


def ConstructModel(
    graph: str,
    graphname: str,
    output_path="Graphs Generated\\Configuration Models Graph Distribution",
) -> nx.graph.Graph:
    """
    This function generates a configuration model based on the degree distribution of the given graph.

    Args:
        graph: The name of the graph from which to take the distribution
        graphname: the name of the generated graph (configuration model)
        output_path: subfolders, following the naming conventions

    Returns:
        The generated Graph and the chosen name
    """

    G = nx.read_edgelist(f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int)
    CM = nx.configuration_model([G.degree[node] for node in G.nodes])

    # Conversion necessary for clustering coefficent function later on
    CM = nx.graph.Graph(CM)

    nx.write_edgelist(
        CM, f"{Project_Path}\\{output_path}\\{graphname}.edgelist", data=False
    )
    print(f"Generated {Project_Path}\\{output_path}\\{graphname}.edgelist")
    return CM, graphname


def ClusteringDistributionPlot(
    G: nx.graph.Graph,
    graphname: str,
    path_add="Configuration Models",
    show: bool = False,
):
    """
    A function to visualize the distribution of the clustering coefficient

    Args:
        G: the graph itself
        graphname: the name of the graph (necessary for the plot)
        path_add: subfolders (default: Configuration Models)
        show: Whether or not the generated plots should be displayed during execution

    Returns:
        Nothing - Plots will be saved
    """
    output_path = f"{Project_Path}\\Plots\\{path_add}\\Clustering\\Distribution"
    C = nx.clustering(G)
    C_val = C.values()
    plt.hist(C_val, bins=np.linspace(0, 1, 200))
    plt.xlabel("Clustering Coefficent")
    plt.ylabel("Number of Instances")
    plt.yscale("log")
    plt.title(f"Clustering {graphname}")
    plt.savefig(f"{output_path}\\Clustering_{graphname}.png")
    print(f"Generated {output_path}\Clustering_{graphname}.png")
    if show:
        plt.ion()
        plt.show(block=True)
    else:
        plt.ioff()
    plt.clf()
    return


def ClusteringDegreePlot(
    G: nx.graph.Graph,
    graphname: str,
    path_add="Configuration Models",
    show: bool = False,
):
    """
    A function to visualize the clustering coefficient against the degree

    Args:
        G: the graph itself
        graphname: the name of the graph (necessary for the plot)
        path_add: subfolders (default: Configuration Models)
        show: Whether or not the generated plots should be displayed during execution

    Returns:
        Nothing - Plots will be saved
    """
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
    plt.title(f"Clustering Degree {graphname}")
    plt.savefig(f"{output_path}\\Clustering-Degree_{graphname}.png")
    print(f"Generated {output_path}\Clustering-Degree_{graphname}.png")
    if show:
        plt.ion()
        plt.show(block=True)
    else:
        plt.ioff()
    plt.clf()
    return


def GenerateFiles(
    graphname: str,
    path_add="Graphs Generated\\Configuration Models Graph Distribution",
    methods: List[str] = ["Abra", "DIAM", "RAND2"],
    GenerateExact: bool = True,
):
    """
    This function generates the approximated betweenness centrality values based on the given graph.


    Args:
        graphname: the name of the graph (used for loading the graph)
        path_add: subfolders (default: Graphs Generated\\Configuration Models Graph Distribution)
        methods: methods to be used for approximation
        GenerateExact: whether the exact values should be calculated (this can take some time)

    Returns:
        Nothing - Files will be saved
    """

    # List of implemented methods. If you add a methods, please also add naming here
    HelperFunctions.MethodComparison(["Abra", "DIAM", "RAND2"], methods)

    G = nk.readGraph(
        f"{Project_Path}\\{path_add}\\{graphname}.edgelist", nk.Format.EdgeListSpaceZero
    )
    # Not fully implemented #TODO
    if GenerateExact:
        approx.Betweenness(G, graphname, True, folder=path_add + "\\Exact_Betweenness")

    if "RAND2" in methods:
        approx.RAND2(G, graphname, True, folder=path_add + "\\Approx_Betweenness")
    if "DIAM" in methods:
        approx.DIAM(G, graphname, folder=path_add + "\\Approx_Betweenness")
    if "Abra" in methods:
        approx.Abra(G, graphname, folder=path_add + "\\Approx_Betweenness")

    return


def ErrorPlot(
    G: nx.graph.Graph,
    methods: List[str],
    graphname: str,
    relative: bool = True,
    path_add="Graphs Generated\\Configuration Models Graph Distribution",
    output_dir_add="Configuration Models",
    show: bool = False,
    log: bool = False,
):
    """
    This function visualizes the betweenness centrality values against the error.

    Args:
        G: the graph itself
        methods: methods to be used for approximation
        relative: whether the betweenness centrality should be displayed absolutely or realtively
        path_add: subfolders (default: Graphs Generated\\Configuration Models Graph Distribution)
        output_dir_add: subfolders for the outputs (default: "Configuration Models")
        show: Whether or not the generated plots should be displayed during execution
        log: Should the plot have a logarithmic x-axis?

    Returns:
        Nothing - All outputs can be found in the Plots folder

    """

    for method in methods:
        # Load error file
        error = IO.file_to_dict(
            f"{Project_Path}\\{path_add}\\Errors\\Error_{graphname}_{method}.txt"
        )
        exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness"
        exact = IO.file_to_dict(f"{exact_path}\\Exact_{graphname}_Abs_norm_True.txt")

        all_bcval = []
        all_errorval = []
        for node in G.nodes:
            all_bcval.append(exact[node])
            if relative:
                all_errorval.append(
                    error[node] / exact[node] if exact[node] != 0 else 0
                )
            else:
                all_errorval.append(error[node])

        plt.title(f"Error - NBC {graphname} {method}")
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
            plt.savefig(f"{output_path}\\BC-Error_{graphname}_{method}_log.png")
            print(f"Generated {output_path}\\BC-Error_{graphname}_{method}_log.png")
        else:
            plt.savefig(f"{output_path}\\BC-Error_{graphname}_{method}.png")
            print(f"Generated {output_path}\\BC-Error_{graphname}_{method}.png")

        if show:
            plt.ion()
            plt.show(block=True)
        else:
            plt.ioff()
        plt.clf()
    return


def DegreePlot(
    graphs: List[str],
    methods: List[str],
    path_add="Graphs Generated\\Configuration Models Graph Distribution",
    output_dir_add="Configuration Models",
    Original: bool = False,
):
    """
    This function visualizes the degree values against the error.
    The function uses functions from the plotting libary module.
    Generates plots with different permutations of the following attributes:
        relative/absolute betweenness centrality, logartihmic/absolute scaling

    Args:
        graphs: List with the names of the graphs which should be used
        methods: list of methods to be used
        path_add: subfolders (default: Graphs Generated\\Configuration Models Graph Distribution)
        output_dir_add: subfolders for the outputs (default: "Configuration Models")
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?

    Returns:
        Nothing - All outputs can be found in the Plots folder
    """

    # Different Permutations of attributes relative and log
    PlottingLibary.plot_error_degree(
        graphs=graphs,
        methods=methods,
        relative=True,
        log=True,
        path_add=path_add,
        output_dir_add=output_dir_add,
        Original=Original,
    )
    PlottingLibary.plot_error_degree(
        graphs=graphs,
        methods=methods,
        relative=True,
        log=False,
        path_add=path_add,
        output_dir_add=output_dir_add,
        Original=Original,
    )
    PlottingLibary.plot_error_degree(
        graphs=graphs,
        methods=methods,
        relative=False,
        log=True,
        path_add=path_add,
        output_dir_add=output_dir_add,
        Original=Original,
    )
    PlottingLibary.plot_error_degree(
        graphs=graphs,
        methods=methods,
        relative=False,
        log=False,
        path_add=path_add,
        output_dir_add=output_dir_add,
        Original=Original,
    )
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
    Generate: bool = True,
    PlotClustering: bool = True,
    PlotError: bool = True,
    PlotErrorDegree: bool = True,
    PlotClusterDegree: bool = True,
    path_add="Graphs Generated\\Configuration Models Graph Distribution",
    methods: List[str] = ["Abra", "DIAM", "RAND2"],
):
    """
    Links the functions of the module and creates a pipeline with which configuration models can be generated, applied and visualized.

    Args:
        graphs: list of graphs which should be used
        Generate: Activate the generation part
        PlotClustering: Enable plotting of clustering
        PlotError: Enable plotting of error plots
        PlotErrorDegree: Enable plotting of error-degree plots
        PlotClusterDegree: Enable plotting of error-clustering plots
        path_add: subfolders (default: Graphs Generated\\Configuration Models Graph Distribution)
        methods: list of methods to be used

    Returns:
        Nothing - but all important files will be saved locally (see terminal outputs)
    """
    if Generate:
        newnames = []
        for graph in graphs:
            graphname = f"ConfigurationModel-{graph}-Distribution"
            newnames.append(graphname)
            newGraph, name = ConstructModel(graph, graphname)
            if PlotClustering:
                ClusteringDistributionPlot(newGraph, name)
            GenerateFiles(name, methods=methods)
            if PlotClusterDegree:
                ClusteringDegreePlot(newGraph, name)
        # Structure inhereted from function
        GenerateErrorFiles.main(
            graphs=newnames,
            Original=False,
            path_add=path_add,
        )
        if PlotErrorDegree:
            DegreePlot(graphs=newnames, methods=methods)
        if PlotError:
            for graph in graphs:
                graphname = f"ConfigurationModel-{graph}-Distribution"
                G = nx.read_edgelist(
                    f"{Project_Path}\\{path_add}\\{graphname}.edgelist", nodetype=int
                )
                ErrorPlot(
                    G, methods=methods, graphname=graphname, relative=True, log=True
                )
                ErrorPlot(
                    G, methods=methods, graphname=graphname, relative=True, log=False
                )
                ErrorPlot(
                    G, methods=methods, graphname=graphname, relative=False, log=True
                )
                ErrorPlot(
                    G, methods=methods, graphname=graphname, relative=False, log=False
                )
    else:
        if PlotClustering or PlotError or PlotClusterDegree:
            for graph in graphs:
                graphname = f"ConfigurationModel-{graph}-Distribution"
                path = f"{Project_Path}\\{path_add}\\{graphname}.edgelist"
                G = nx.read_edgelist(path, nodetype=int)
                if PlotClustering:
                    ClusteringDistributionPlot(G, graphname)
                if PlotClusterDegree:
                    ClusteringDegreePlot(G, graphname)
                if PlotError:
                    ErrorPlot(
                        G, methods=methods, graphname=graphname, relative=True, log=True
                    )
                    ErrorPlot(
                        G,
                        methods=methods,
                        graphname=graphname,
                        relative=True,
                        log=False,
                    )
                    ErrorPlot(
                        G,
                        methods=methods,
                        graphname=graphname,
                        relative=False,
                        log=True,
                    )
                    ErrorPlot(
                        G,
                        methods=methods,
                        graphname=graphname,
                        relative=False,
                        log=False,
                    )
        if PlotErrorDegree:
            newnames = []
            for graph in graphs:
                graphname = f"ConfigurationModel-{graph}-Distribution"
                newnames.append(graphname)
            DegreePlot(graphs=newnames, methods=methods)

    return


if __name__ == "__main__":
    sys.exit(
        main(
            ["ca-GrQc", "ca-HepPh", "ca-HepTh"],
            Generate=False,
            PlotClustering=False,
            PlotError=False,
            # PlotErrorDegree=False,
            PlotClusterDegree=False,
        )
    )
