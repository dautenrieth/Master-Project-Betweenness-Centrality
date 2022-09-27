"""
Collection of Input/Output Functions used in other modules
"""

from typing import List, Optional, Tuple, Dict


def save_in_file(
    projectpath,
    scores: List,
    graph_name: str,
    method_name: str,
    normal: bool = False,
    typename: str = "Approx",
    folder: str = "Approx_Betweenness",
):
    """
    Saving List into file.
    Format:
        1: score1
        2: score2
        etc.

    Args:
        projectpath: the standard path to your project
        scores: the data you want to store
        graph_name: name of the graph (used in the filenaming)
        method_name: name of the method (used in the filenaming)
        normal: if values are normalized (used in filenaming)
        typename: typename of data (default: "Approx")
        folder: name of the folder you want to save the file in

    Returns:
        None - File will be saved locally
    """

    # Opening a file, writing to it, and then closing it.
    with open(
        f"{projectpath}/{folder}/{typename}_{graph_name}_{method_name}_norm_{normal}.txt",
        "w",
    ) as fp:
        fp.write(
            "\n".join("{}:    {}".format(node, x) for node, x in enumerate(scores))
        )
    print(
        f"Generated {projectpath}\{folder}\{typename}_{graph_name}_{method_name}_norm_{normal}.txt"
    )
    return


def file_to_dict(path: str) -> Dict[int, float]:
    """
    Reads back a file to a dictionary.
    Used for files saved with the save_in_file- or dic_to_file-function.

    Args:
        path: the path to the file

    Returns:
        A dictionary with the key being an integer and the value being a float.
    """
    dictionary = {}
    with open(path) as file:
        for line in file:
            (key, value) = line.split(":", 1)

            dictionary[int(key)] = float(value)
    return dictionary


def dic_to_file(
    projectpath,
    dic: dict,
    filename: str,
    folder: str = "Regression\\Predictions",
):
    """
    Used to save dictionary style data to a file.

    Args:
        projectpath: the standard path to your project
        dic: the dictionary you want to save
        filename: naming of the file
        folder: name of the folder you want to save the file in (default: "Regression\\Predictions")

    Retuns:
        None - File will be saved locally
    """
    with open(
        f"{projectpath}/{folder}/{filename}.txt",
        "w",
    ) as fp:
        fp.write("\n".join("{}:    {}".format(node, x) for node, x in dic.items()))
    print(f"Generated {projectpath}/{folder}/{filename}.txt")
    return
