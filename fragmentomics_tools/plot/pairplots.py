import numpy as numpy

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.cm as cm


def make_interquartile_boxes(
    ax, X, Y, facecolor=None, edgecolor="None", alpha=0.1, s=10
):
    """
    Creates inter-quartile boxes for X and Y
    :param ax:
    :param X: np.array (n_obs, 3). n_obs is the number of, e.g., cell types. The three columns
     give the lower bound, the center, and the upper bound of the estimate. This can be the
     quartiles or mean with standard deviation or anything else
    :param Y: np.array (n_obs, 3)
    :param facecolor: facecolor of the boxes
    :param edgecolor: edge color of the boxes
    :param alpha: opacity of the boxes
    :return:
    """

    assert X.shape == Y.shape, f"Input arrays must be the same shape."
    assert len(X.shape) == 2, f"Input arrays must be 2D"
    assert (
        X.shape[1] == 3
    ), f"Input arrays must have 3 columns, 1st quartile, median, 3rd quartile"

    errorboxes = [
        Rectangle((x[0], y[0]), x[2] - x[0], y[2] - y[0]) for x, y in zip(X, Y)
    ]
    if facecolor is None:
        facecolor = [cm.Set3(i / len(errorboxes)) for i, _ in enumerate(errorboxes)]

    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )
    ax.add_collection(pc)

    xerr = numpy.abs(X - X[:, 1][:, None])[:, [0, 2]]
    yerr = numpy.abs(Y - Y[:, 1][:, None])[:, [0, 2]]

    ax.scatter(X[:, 1], Y[:, 1], color="k", s=s)
    _ = ax.errorbar(
        X[:, 1], Y[:, 1], xerr=xerr.T, yerr=yerr.T, fmt="None", ecolor="k", alpha=0.5
    )

    return ax
