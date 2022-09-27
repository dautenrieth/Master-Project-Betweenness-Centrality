"""
This module is used to run different types of approximation methods
This is a variation to the RunAppproximations module for using generated graphs.
Due to naming conventions and selections options the modification was
neceassry - in the future, however, these modules can be merged
"""

import networkit as nk
from typing import List, Optional, Tuple, Dict
import sys
import logging
import time
import datetime
from pathlib import Path
import IO
from os import listdir
from os.path import isfile, join
import os

Project_Path = os.path.dirname(os.path.abspath(__file__))


def Abra(Graph_Path: Path, G: nk.graph.Graph, graph_name: str, normal: bool = False):
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
        Graph_Path, [x / 2 for x in kadabra.scores()], graph_name, "Abra", True
    )
    return


def RAND2(Graph_Path: Path, G: nk.graph.Graph, graph_name: str, normal: bool = False):
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
    size = G.numberOfNodes() / 10
    est = nk.centrality.EstimateBetweenness(G, size, normalized=normal)
    est.run()
    logging.info(f"{graph_name} - RAND2: {time.perf_counter()-t:0.4f} seconds")
    IO.save_in_file(Graph_Path, est.scores(), graph_name, "RAND2", normal)
    return


def DIAM(
    Graph_Path: Path, G: nk.graph.Graph, graph_name: str, normal: bool = True
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
    IO.save_in_file(Graph_Path, ab.scores(), graph_name, "DIAM", True)
    return


def Betweenness(
    Graph_Path: Path, G: nk.graph.Graph, graph_name: str, normal: bool = False
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
        Graph_Path,
        abso.scores(),
        graph_name,
        "Abs",
        normal,
        typename="Exact",
        folder="Exact_Betweenness",
    )
    return


def run(Graph_Path: Path, Exact=False):
    """
    This function will initalize logging and the run entered approximation methods.
    All graphs in the passed folder path will be used.

    Args:
        Graph_Path: the path to the folder where the graphs are loacted
        Exact: do you want to calculate the exact betweenness centrality values?

    Returns:
        Nothing
    """
    global t
    e = datetime.datetime.now()
    logging.basicConfig(
        filename=f'{Graph_Path}/Approx_Betweenness/logs/run_{e.strftime("%Y-%m-%d_%H%M")}.log',
        level=logging.INFO,
    )
    formatter = logging.Formatter("%(message)s")
    logging.info("Started")
    graphs = [f for f in listdir(Graph_Path) if isfile(join(Graph_Path, f))]
    print(graphs)
    for graph in graphs:

        # Read graph
        G = nk.readGraph(f"{Graph_Path}/{graph}", nk.Format.EdgeListSpaceZero)

        name = graph.split(".", 1)[0]
        print(name)
        print(G)

        RAND2(Graph_Path, G, name, True)

        DIAM(Graph_Path, G, name)

        Abra(Graph_Path, G, name)

        if Exact:
            Betweenness(Graph_Path, G, name, normal=True)


def main():
    """
    Function for selecting different types of generated graphs and then calling the
    run-functions with the specified paths.

    Args:
        None - All attributes are hardcoded atm

    Returns:
        Nothing
    """
    # Select fpr which graphs you want to run approximations and exact values
    Configurationmodel_nconstant = False
    Configurationmodel_dconstant = True
    ErdosRenyi = False

    if Configurationmodel_nconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant n"
        )
        run(
            Path(f"{Project_Path}\\Graphs Generated\\Configuration Model\\n constant"),
            Exact=True,
        )

    if Configurationmodel_dconstant:
        print(
            "Started to run Approximations for Configuration model graphs with constant degree"
        )
        run(
            Path(f"{Project_Path}\\Graphs Generated\\Configuration Model\\d constant"),
            Exact=True,
        )

    if ErdosRenyi:
        print("Started to run Approximations for Erdos Renyi Graphs")
        run(Path(f"{Project_Path}\\Graphs Generated\\Erdos Renyi"), Exact=True)
    return


if __name__ == "__main__":
    sys.exit(main())
