"""
This module is used to run different types of approximation methods
"""

import networkit as nk
import os
from typing import List, Optional, Tuple, Dict
import sys
import logging
import time
import datetime
import IO
import HelperFunctions

# Project File Path
Project_Path = os.path.dirname(os.path.abspath(__file__))

t = time.perf_counter()


def Abra(
    G: nk.graph.Graph,
    graph_name: str,
    normal: bool = False,
    folder: str = "Approx_Betweenness",
):
    """
    This function runs the Kadabra approximation method. The naming is based on the paper:
    "A Benchmark for Betweenness Centrality Approximation Algorithms on Large Graphs". In this case the naming
    refers to the type of approximation used not the exact algorithm.
    Runtime will be logged.

    Args:
        G: the graph to be used
        graph_name: the name of the graph
        normal: this function is obsolete for this method because all predictions will be normalized automatically
        folder: the name of the folder where the approximations should be saved (default: "Approx_Betweenness")

    Returns:
        a list of all approximated values sorted by node number
    """
    t = time.perf_counter()
    kadabra = nk.centrality.KadabraBetweenness(G, 0.01, 0.1)
    kadabra.run()
    logging.info(f"{graph_name} - Abra: {time.perf_counter()-t:0.4f} seconds")
    # Need to divide scores by 2 because of used normalization factor
    IO.save_in_file(
        Project_Path,
        [x / 2 for x in kadabra.scores()],
        graph_name,
        "Abra",
        True,
        folder=folder,
    )
    return [x / 2 for x in kadabra.scores()]


def RAND2(
    G: nk.graph.Graph,
    graph_name: str,
    normal: bool = False,
    folder: str = "Approx_Betweenness",
):
    """
    This function runs the an approximation method here called RAND2. The naming is based on the paper:
    "A Benchmark for Betweenness Centrality Approximation Algorithms on Large Graphs". In this case the naming
    refers to the type of approximation used not the exact algorithm.
    Runtime will be logged.

    Args:
        G: the graph to be used
        graph_name: the name of the graph
        normal: Should the approximated values be normalized?
        folder: the name of the folder where the approximations should be saved (default: "Approx_Betweenness")

    Returns:
        a list of all approximated values sorted by node number
    """
    t = time.perf_counter()
    size = G.numberOfNodes() / 15
    est = nk.centrality.EstimateBetweenness(G, size, normalized=normal)
    est.run()
    logging.info(f"{graph_name} - RAND2: {time.perf_counter()-t:0.4f} seconds")
    IO.save_in_file(
        Project_Path, est.scores(), graph_name, "RAND2", normal, folder=folder
    )
    return est.scores()


def DIAM(
    G: nk.graph.Graph,
    graph_name: str,
    normal: bool = True,
    folder: str = "Approx_Betweenness",
):  # path-sampling
    """
    This function runs the an approximation method here called DIAM. The naming is based on the paper:
    "A Benchmark for Betweenness Centrality Approximation Algorithms on Large Graphs". In this case the naming
    refers to the type of approximation used not the exact algorithm.
    Runtime will be logged.

    Args:
        G: the graph to be used
        graph_name: the name of the graph
        normal: Should the approximated values be normalized?
        folder: the name of the folder where the approximations should be saved (default: "Approx_Betweenness")

    Returns:
        a list of all approximated values sorted by node number
    """
    t = time.perf_counter()
    ab = nk.centrality.ApproxBetweenness(G, epsilon=0.01)
    ab.run()
    logging.info(f"{graph_name} - DIAM: {time.perf_counter()-t:0.4f} seconds")
    IO.save_in_file(Project_Path, ab.scores(), graph_name, "DIAM", True, folder=folder)
    return ab.scores()


# This function doesnt need to be used for original graphs where the exact values are already calculated
def Betweenness(
    G: nk.graph.Graph,
    graph_name: str,
    normal: bool = False,
    folder: str = "Approx_Betweenness",
):
    """
    This function will calculate the exact betweenness centrality value.
    This can take a while based on the size of input graph.
    Runtime will be logged.

    Args:
        G: the graph to be used
        graph_name: the name of the graph
        normal: Should the approximated values be normalized?
        folder: the name of the folder where the approximations should be saved (default: "Approx_Betweenness")

    Returns:
        a list the exact values sorted by node number
    """
    t = time.perf_counter()
    abso = nk.centrality.Betweenness(G, normalized=normal)
    abso.run()
    logging.info(f"{graph_name} - Betweenness: {time.perf_counter()-t:0.4f} seconds")
    IO.save_in_file(
        Project_Path,
        abso.scores(),
        graph_name,
        "Abs",
        normal,
        typename="Exact",
        folder=folder,
    )
    return abso.scores()


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
    This main function will initalize logging and the run entered approximation methods

    Args:
        graphs: list of graphs which should be used
        methods: methods to be used for approximation

    Returns:
        Nothing
    """
    global t
    e = datetime.datetime.now()
    logging.basicConfig(
        filename=f'{Project_Path}/Approx_Betweenness/logs/run_{e.strftime("%Y-%m-%d_%H%M")}.log',
        level=logging.INFO,
    )
    formatter = logging.Formatter("%(message)s")
    logging.info("Started")
    for graph in graphs:

        # Read graph
        G = nk.readGraph(
            f"{Project_Path}\\Graphs\\{graph}.lcc.net", nk.Format.EdgeListSpaceZero
        )

        # List of implemented methods. If you add a methods, please also add naming here
        HelperFunctions.MethodComparison(["Abra", "DIAM", "RAND2"], methods)

        # Check if selected Methods are one of the implemented methods

        if "RAND2" in methods:
            RAND2(G, graph, True)
        if "DIAM" in methods:
            DIAM(G, graph)
        if "Abra" in methods:
            Abra(G, graph)


if __name__ == "__main__":
    sys.exit(main())
