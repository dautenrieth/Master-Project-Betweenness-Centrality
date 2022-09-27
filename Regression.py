"""
This module was used to create and train different regression models.
The data used to train models depend on specific approximation runs.
If you run the approximations again and you want to use model number 2,
please adjust the number of sampled paths according to the calculated
number of the GetNodeBranches-module.
All indiviudal steps will be saved locally. So when running this module
again only run necessary steps to save time.
Saved models and steps can be found in the Regression folder.
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
import json
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from typing import List, Optional, Tuple, Dict
import pickle
import warnings

import IO
import PlottingLibary

warnings.simplefilter(action="ignore", category=FutureWarning)

# Define which graphs are used
graphs = ["ca-GrQc", "ca-HepPh", "ca-HepTh"]
# Those numbers are calculated in the GetNodeBranches.py file and have to be adjusted after running approximations again
SampledPaths = {"ca-GrQc": 35964, "ca-HepPh": 28584, "ca-HepTh": 17640}
Project_Path = os.path.dirname(os.path.abspath(__file__))
method = "Abra"
output_folder = "Regression"

# The factors listed below will be used in the regression models
# Modelnumber 1
feature_names = [
    "ApproxValue",
    "maxDegree",
    "NumberNodes",
    "NumberEdges",
    "Degree",
    "ClusteringCoeff",
]

# Modelnumber 2
feature_names2 = [
    "TimesSampled",
    "maxDegree",
    "NumberNodes",
    "NumberEdges",
    "Degree",
    "ClusteringCoeff",
    "SampledPaths",
]

# Modelnumber 3
feature_names3 = [
    "Diameter",
    "maxDegree",
    "NumberNodes",
    "NumberEdges",
    "Degree",
    "ClusteringCoeff",
]

# Modelnumber 4
feature_names4 = [
    "maxDegree",
    "NumberNodes",
    "NumberEdges",
    "Degree",
    "ClusteringCoeff",
    "DegreeCentralitiy",
]


def ConstructData(Modelnumber: int, Original: bool = True, path_add=""):
    """
    This function creates the data table necessary for training the regression model.
    Args:
        Modelnumber: Choose from the implemented options (See feature_name list)
        Original: Are the graphs in the list the ones defined as original or generated graphs?
        path_add: existing subfolders of the data

    Returns:
        DataFrame with training data - also saves data locally

    """
    if Modelnumber == 1:
        df = pd.DataFrame(columns=feature_names + ["ExactValue"])
    elif Modelnumber == 2:
        df = pd.DataFrame(columns=feature_names2 + ["ExactValue"])
    elif Modelnumber == 3:
        df = pd.DataFrame(columns=feature_names3 + ["ExactValue"])
    elif Modelnumber == 4:
        df = pd.DataFrame(columns=feature_names4 + ["ExactValue"])
    else:
        raise Exception("Modelnumber not implemented")
    i = 0
    for graph in graphs:
        print(f"Process {graph}")
        # Read graph
        G = nx.read_edgelist(f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int)
        C = nx.clustering(G)
        maxDegree = max([elem[1]] for elem in list(G.degree))
        if Original:
            exactname = f"{graph}.txt"
            exact_path = (
                f"{Project_Path}\\{path_add}\\Exact_Betweenness\\Normalized_Scores"
            )
        else:
            exactname = f"Exact_{graph}_Abs_norm_True.txt"
            exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness"
        exact = IO.file_to_dict(f"{exact_path}\\{exactname}")

        approx_path = f"{Project_Path}\\{path_add}\\Approx_Betweenness"
        approxname = f"Approx_{graph}_{method}_norm_True.txt"
        approx = IO.file_to_dict(f"{approx_path}\\{approxname}")

        numberEdges = G.number_of_edges()
        numberNodes = G.number_of_nodes()
        if Modelnumber == 3:
            diameter = nx.diameter(G)

        if Modelnumber == 4:
            degree_centr = nx.degree_centrality(G)

        for node in G.nodes:
            if Modelnumber == 1:
                new_row = pd.DataFrame(
                    {
                        "ApproxValue": approx[node],
                        "maxDegree": maxDegree,
                        "NumberNodes": numberNodes,
                        "NumberEdges": numberEdges,
                        "Degree": G.degree[node],
                        "ClusteringCoeff": C[node],
                        "ExactValue": exact[node],
                    }
                )
            elif Modelnumber == 2:
                new_row = pd.DataFrame(
                    {
                        "TimesSampled": SampledPaths[graph] * approx[node],
                        "maxDegree": maxDegree,
                        "NumberNodes": numberNodes,
                        "NumberEdges": numberEdges,
                        "Degree": G.degree[node],
                        "ClusteringCoeff": C[node],
                        "SampledPaths": SampledPaths[graph],
                        "ExactValue": exact[node],
                    }
                )
            elif Modelnumber == 3:
                new_row = pd.DataFrame(
                    {
                        "Diameter": diameter,
                        "maxDegree": maxDegree,
                        "NumberNodes": numberNodes,
                        "NumberEdges": numberEdges,
                        "Degree": G.degree[node],
                        "ClusteringCoeff": C[node],
                        "ExactValue": exact[node],
                    }
                )
            elif Modelnumber == 4:
                new_row = pd.DataFrame(
                    {
                        "maxDegree": maxDegree,
                        "NumberNodes": numberNodes,
                        "NumberEdges": numberEdges,
                        "Degree": G.degree[node],
                        "ClusteringCoeff": C[node],
                        "DegreeCentralitiy": degree_centr[node],
                        "ExactValue": exact[node],
                    }
                )
            df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)
        result = df.to_json(orient="table")

    # Save create Dataframe
    with open(f"{output_folder}\\RegressionData.json", "w") as f:
        json.dump(result, f)
    print(f"Generated {output_folder}\\RegressionData.json")
    return df


def LoadDf():
    """
    This function loads the in json format saved Dataframe and returns it.
    Only works with standard naming and folders

    Args:
        None

    Returns:
        DataFrame with training data
    """
    with open(f"{output_folder}\\RegressionData.json", "r") as f:
        read = json.load(f)
    df = pd.read_json(read, orient="table")
    print(f"Loaded {output_folder}\\RegressionData.json")
    return df


def RegressionModelTrain(X, y):
    """
    This function trains the regression model

    Args:
        X: training data
        y: true values

    Returns:
        Fitted regression model - also saves it locally
    """
    mod = sm.OLS(y.astype(float), X.astype(float))
    res = mod.fit()
    print(res.summary())
    with open(f"{output_folder}\\RM_summary.csv", "w") as fh:
        fh.write(res.summary().as_csv())
    res.save(f"{output_folder}\\RegressionModel.pickle")
    return res


def RegressionModelTrainwithInteractions(X, y):
    """
    This function trains the regression model. It also extends
    the standard model and adds interactions between variables

    Args:
        X: training data
        y: true values

    Returns:
        Fitted regression model - also saves it locally
    """
    poly = PolynomialFeatures(interaction_only=True)
    X_tr = poly.fit_transform(X)
    Xt = pd.concat(
        [
            X,
            pd.DataFrame(X_tr, columns=poly.get_feature_names()).drop(
                ["1", "x0", "x1", "x2", "x3", "x4"], 1
            ),
        ],
        1,
    ).astype(float)

    mod = sm.OLS(y.astype(float), Xt.astype(float))
    res = mod.fit()
    print(res.summary())
    with open(f"{output_folder}\\RMI_summary.csv", "w") as fh:
        fh.write(res.summary().as_csv())
    res.save(f"{output_folder}\\RegressionModelwInteractions.pickle")
    return res


def loadModel(modelname: str):
    """
    This function can be used to load the fitted regression model.
    Input should be pickle format

    Args:
        modelname: the name of the model with ending (e.g. RegressionModel.pickle)

    Returns:
        Fitted Regression model

    Exceptions:
        Throws Exception when no model is available
    """
    if modelname in [file for file in os.listdir(f"{Project_Path}\\{output_folder}")]:
        print(f"Loaded {modelname}")
        return pickle.load(open(f"{output_folder}\\{modelname}", "rb"))
    else:
        raise Exception(f"No Regression Model with the name {modelname} found")


def createPredictionFiles(
    Model,
    Modelnumber: int,
    graph: str,
    withInteractions: bool = False,
    Original: bool = True,
    path_add="",
):
    """
    This function can be used to predict data and save the predictions

    Args:
        Model: the fitted regression model
        Modelnumber: the type of the model (see feature_name list)
        graph: the graph which was used for training
        withInteractions: Whether the model includes interactions between variables
        Original: Are the graphs in the list the ones defined as original or generated graphs?
        path_add: existing subfolders of the data

    Returns:
        Nothing - predictions will be saved locally
    """
    if Original:
        G = nx.read_edgelist(f"{Project_Path}\\Graphs\\{graph}.lcc.net", nodetype=int)
    else:
        G = nx.read_edgelist(
            f"{Project_Path}\\{path_add}\\{graph}.edgelist", nodetype=int
        )
    C = nx.clustering(G)
    maxDegree = max([elem[1]] for elem in list(G.degree))

    approx_path = f"{Project_Path}\\{path_add}\\Approx_Betweenness"
    approxname = f"Approx_{graph}_{method}_norm_True.txt"
    approx = IO.file_to_dict(f"{approx_path}\\{approxname}")

    numberEdges = G.number_of_edges()
    numberNodes = G.number_of_nodes()

    if Modelnumber == 3:
        diameter = nx.diameter(G)

    if Modelnumber == 4:
        degree_centr = nx.degree_centrality(G)

    pred = {}
    for node in range(numberNodes):
        if Modelnumber == 1:
            new_row = pd.DataFrame(
                {
                    "ApproxValue": approx[node],
                    "maxDegree": maxDegree,
                    "NumberNodes": numberNodes,
                    "NumberEdges": numberEdges,
                    "Degree": G.degree[node],
                    "ClusteringCoeff": C[node],
                }
            )
        elif Modelnumber == 2:
            new_row = pd.DataFrame(
                {
                    "TimesSampled": SampledPaths[graph] * approx[node],
                    "maxDegree": maxDegree,
                    "NumberNodes": numberNodes,
                    "NumberEdges": numberEdges,
                    "Degree": G.degree[node],
                    "ClusteringCoeff": C[node],
                    "SampledPaths": SampledPaths[graph],
                }
            )
        elif Modelnumber == 3:
            new_row = pd.DataFrame(
                {
                    "Diameter": diameter,
                    "maxDegree": maxDegree,
                    "NumberNodes": numberNodes,
                    "NumberEdges": numberEdges,
                    "Degree": G.degree[node],
                    "ClusteringCoeff": C[node],
                }
            )
        elif Modelnumber == 4:
            new_row = pd.DataFrame(
                {
                    "maxDegree": maxDegree,
                    "NumberNodes": numberNodes,
                    "NumberEdges": numberEdges,
                    "Degree": G.degree[node],
                    "ClusteringCoeff": C[node],
                    "DegreeCentralitiy": degree_centr[node],
                }
            )
        if withInteractions:
            if Modelnumber == 1:
                features = feature_names
            elif Modelnumber == 2:
                features = feature_names2
            elif Modelnumber == 3:
                features = feature_names3
            elif Modelnumber == 4:
                features = feature_names4
            else:
                raise Exception("Modelnumber not implemented")
            Xb = new_row[features]
            poly = PolynomialFeatures(interaction_only=True)
            X_tr = poly.fit_transform(Xb)
            Xt = pd.concat(
                [
                    Xb,
                    pd.DataFrame(X_tr, columns=poly.get_feature_names()).drop(
                        ["1", "x0", "x1", "x2", "x3", "x4"], 1
                    ),
                ],
                1,
            )
            pred_data = Xt
        else:
            pred_data = new_row

        # pred_data = np.array(pred_data, dtype=float)
        p = Model.predict(pred_data)[0]
        if p >= 0:
            pred[node] = p
        else:
            pred[node] = 0.0
    if withInteractions:
        name = f"RMI_Predictions_{graph}_{method}"
    else:
        name = f"RM_Predictions_{graph}_{method}"
    IO.dic_to_file(projectpath=Project_Path, dic=pred, filename=name)
    return


def GenerateErrorFiles(
    graphs: List[str] = [
        "ca-GrQc",
        "email-Enron",
        "ca-HepTh",
        "ca-HepPh",
        "com-amazon",
        "com-lj",
        "dbpedia-link",
    ],
    Original: bool = True,
    path_add: str = "Regression",
    path_add_exact: str = "",
    approx_foldername="Predictions",
):
    """
    This function will calculate the error of the predictions and exact values
    and will save the results locally.

    Args:
        graphs: list of graphs which should be used
        Orignal: Are the graphs in the list the ones defined as original or generated graphs?
        path_add: existing subfolders of the data
        path_add_exact: the subfolder of the exact betweenness centrality data
        approx_foldername: the foldername where the approximation data can be found

    Returns:
        Nothing - File will be saved locally
    """
    if Original:
        exact_path = (
            f"{Project_Path}\\{path_add_exact}\\Exact_Betweenness\\Normalized_Scores"
        )
    else:
        exact_path = f"{Project_Path}\\{path_add}\\Exact_Betweenness"
    approx_path = f"{Project_Path}\\{path_add}\\{approx_foldername}"
    error_path = f"{Project_Path}\\{path_add}\\Errors"

    approx_files = [f for f in listdir(approx_path) if isfile(join(approx_path, f))]
    exact_files = [f for f in listdir(exact_path) if isfile(join(exact_path, f))]

    for file in approx_files:

        split_string = file.split("_", 3)
        modeltype = split_string[0]
        graphname = split_string[2]
        methodname = split_string[3].replace(".txt", "")

        # Get all normalized files and the according exact values
        if (
            (modeltype == "RM" or modeltype == "RMI")
            and graphname in graphs
            and methodname == method
        ):

            if Original:
                filename = f"{graphname}.txt"
            else:
                filename = f"Exact_{graphname}_Abs_norm_True.txt"

            if f"{filename}" in exact_files:
                exact = IO.file_to_dict(f"{exact_path}\\{filename}")
                approx = IO.file_to_dict(f"{approx_path}\\{file}")

                # Calculate absolute error
                error = {key: exact[key] - approx.get(key, 0) for key in exact.keys()}

                # Write Error Files
                with open(
                    f"{error_path}\\Error_{graphname}_{method}_{modeltype}.txt",
                    "w",
                ) as fp:
                    fp.write(
                        "\n".join(
                            "{}:    {}".format(node, x) for node, x in error.items()
                        )
                    )
                print(
                    f"Generated {error_path}\\Error_{graphname}_{method}_{modeltype}.txt"
                )

            else:
                raise Exception(f"{filename} seems to be missing")
    return


def main(
    Modelnumber: int = 1,
    createDataset: bool = False,
    trainModels: bool = False,
    withInteraction: bool = True,
    createPredictions: bool = True,
    createError: bool = True,
    createErrorPlots: bool = True,
):
    """
    The main function combines all other functions and streamlines them.
    Dependend on the inputs this function can exceute single steps or
    all steps in sequence.

    Args:
        Modelnumber: select the type of model you want to use (see feature_name list)
        createDataset: if True a training dataset will be created otherwise an existing set will be used for training
        trainModels: if True a model will be trained otherwise an existing one will be loeaded
        withInteraction: the model has interactions between variables?
        createPredictions: do you want to create predictions?
        createError: do you want to create error files based on predictions?
        createErrorPlots: do you want to visualize errors?

    Returns:
        Nothing
    """
    if (
        "RegressionData.json"
        in [
            f
            for f in os.listdir(f"{Project_Path}\\{output_folder}")
            if os.path.isfile(os.path.join(f"{Project_Path}\\{output_folder}", f))
        ]
        and not createDataset
    ):
        df = LoadDf()
    else:
        df = ConstructData(Modelnumber=Modelnumber)

    if Modelnumber == 1:
        features = feature_names
    elif Modelnumber == 2:
        features = feature_names2
    elif Modelnumber == 3:
        features = feature_names3
    elif Modelnumber == 4:
        features = feature_names4
    else:
        raise Exception("Modelnumber not implemented")

    if withInteraction:
        if trainModels:
            RMI = RegressionModelTrainwithInteractions(df[features], df["ExactValue"])
        else:
            RMI = loadModel("RegressionModelwInteractions.pickle")
        if createPredictions:
            for graph in graphs:
                createPredictionFiles(
                    RMI, Modelnumber=Modelnumber, graph=graph, withInteractions=True
                )
    else:
        if trainModels:
            RM = RegressionModelTrain(df[features], df["ExactValue"])
        else:
            RM = loadModel("RegressionModel.pickle")
        if createPredictions:
            for graph in graphs:
                createPredictionFiles(RM, Modelnumber=Modelnumber, graph=graph)

    if createError:
        GenerateErrorFiles(
            graphs, path_add="Regression", approx_foldername="Predictions"
        )
    if createErrorPlots:
        for l in [True, False]:
            for r in [True, False]:
                for rm in ["RM", "RMI"]:
                    PlottingLibary.plot_error_bc(
                        graphs=graphs,
                        methods=[method],
                        path_add="Regression",
                        output_dir_add="Regression",
                        relative=r,
                        log=l,
                        Regression=True,
                        Regression_model=rm,
                    )
    return


# Select here which parts of the process you want to execute when runnning this module
if __name__ == "__main__":
    sys.exit(
        main(
            Modelnumber=1,
            createDataset=True,
            withoutInteraction=True,
            trainModels=True,
            # createPredictions=False,
            # createError=False,
            # createErrorPlots=False,
        )
    )


"""
Useful/Used resources:
https://towardsdatascience.com/multiple-linear-regression-with-interactions-unveiled-by-genetic-programming-4cc325ac1b65

"""
