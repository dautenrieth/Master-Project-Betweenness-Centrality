"""
   This module contains several utility functions of the Plotting Libary
"""

import numpy as np


def scatter_hist(x, y, ax, ax_histx, bins):
    """
    A utility function which defines histograms

    Args:
        x: x-axis data
        y: y-axis data
        ax: axis of the scatter plot
        ax_histx: axis of the histogramm
        bins: the created bins

    Returns:
        Nothing directly
    """
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    ymax = np.max(np.abs(y))
    ymax += ymax * 0.1
    ax.set_ylim([-ymax, ymax])
    lim = 1

    bins = bins

    ax_histx.hist(x, bins=bins)
    return


def binning(x, linear=True):
    """
    function that creates the bins

    Args:
        x: data
        linear: If True linear space is used, logartihmic space otherwise

    Returns:
        Bins: list with elements
    """
    if linear:
        return np.linspace(0, max(x), 20)
    else:
        return np.append(
            np.array([0]), (np.logspace(0, np.log(max(x)) / np.log(10), num=20))
        )


def movingaverage(x, y, bins, mult=1):
    """
    function which calculates the moving average

    Args:
        x: x-axis data
        y: y-axis data
        bins: list of bins
        mult: multiplication factor to scale up average (can be used for better visualization)

    Returns:
        list of moving averages
    """
    digitized = np.digitize(x, bins)
    means = [[] for i in range(len(bins))]
    mean_list = []
    for index, b in enumerate(digitized):
        means[b - 1].append(abs(y[index]))
    for i in range(len(bins)):
        # print(f'{i}: {sum(means[i])},{len(means[i])}')
        mean_list.append(
            (sum(means[i]) / len(means[i]) * mult) if len(means[i]) != 0 else 0
        )
    return mean_list
