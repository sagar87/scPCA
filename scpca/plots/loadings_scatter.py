from typing import Callable, List, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from adjustText import adjust_text  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore

from ..utils import state_loadings
from ..utils.data import _validate_sign
from .helper import _annotate_dots, _plot_dots, _rand_jitter, _set_up_cmap, _set_up_plot


def loadings_scatter(
    adata: AnnData,
    model_key: str,
    states: Union[List[str], str] = [],
    factor: Union[int, List[int], None] = None,
    var_names: Union[List[str], str] = [],
    variable: str = "W",
    highlight: bool = True,
    size_func: Callable[[float], float] = lambda x: 10.0,
    sign: Union[int, float] = 1.0,
    jitter: float = 0.01,
    fontsize: int = 10,
    show_labels: Union[List[int], int] = 0,
    annotation_linewidth: float = 0.5,
    connection_linewidth: float = 0.5,
    connection_alpha: float = 1.0,
    cmap: str = "RdBu",
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Axes = None,
) -> Axes:
    """
    Create a scatter plot of loading states for a given model and factors.

    Parameters
    ----------
    adata
        AnnData object.
    model_key
        Key to access the model information in `adata.uns`.
    states
        States to plot compare. Can be more than two.
    factor
        Factor or list of factors to plot. If None, all factors are plotted. Default is None.
    var_names
        Variable names to highlight. Default is an empty list.
    variable
        Variable to plot. Default is "W".
    highlight
        Whether to highlight specific variables. Default is True.
    size_func
        Function to determine the size of the dots. Default is a lambda function that returns 10.0.
    sign
        Sign of the loadings to consider. Default is 1.0.
    jitter
        Amount of jitter to apply to the x-coordinates. Default is 0.01.
    fontsize
        Font size for annotations. Default is 10.
    show_labels
        Indices of states for which to show labels. Default is 0.
    annotation_linewidth
        Line width for annotations. Default is 0.5.
    cmap
        Colormap to use. Default is "RdBu".
    ncols
        Number of columns in the subplot grid. Default is 4.
    width
        Width of each subplot in inches. Default is 4.
    height
        Height of each subplot in inches. Default is 3.
    ax
        Matplotlib axes to use for plotting. Default is None.

    Returns
    -------
    ax
        Matplotlib axes object containing the plotted instances or factors.

    Notes
    -----
    - The function sets up the plot environment and calls the internal function `_loadings_scatter` to do the actual plotting.

    Examples
    --------
    # Example usage
    loadings_scatter(adata, 'm0', ['state1', 'state2'], factor=0)
    """

    _ = _validate_sign(sign)

    if isinstance(var_names, str):
        var_names = [var_names]

    if isinstance(show_labels, (int, str)):
        show_labels = [show_labels]

    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _loadings_scatter,
        states=states,
        var_names=var_names,
        variable=variable,
        sign=sign,
        highlight=highlight,
        jitter=jitter,
        fontsize=fontsize,
        size_func=size_func,
        show_labels=show_labels,
        annotation_linewidth=annotation_linewidth,
        connection_linewidth=connection_linewidth,
        connection_alpha=connection_alpha,
        cmap=cmap,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _loadings_scatter(
    adata: AnnData,
    model_key: str,
    factor: int,
    states: List[str],
    var_names: List[str],
    sign: Union[int, float],
    variable: str,
    highlight: bool,
    size_func: Callable[[float], float],
    jitter: float,
    fontsize: int,
    show_labels: List[int],
    annotation_linewidth: float,
    connection_linewidth: float,
    connection_alpha: float,
    cmap: str,
    ax: Axes = None,
) -> Axes:
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    texts = []
    coords = np.zeros((len(states), len(var_names), 2))

    for i, state in enumerate(states):
        df = state_loadings(
            adata, model_key, state, factor, variable=variable, highest=adata.shape[1], lowest=0, sign=sign
        )

        y = df.weight.values
        x = i * np.ones_like(y)
        x = _rand_jitter(x, jitter * np.abs(y))
        cmap, norm = _set_up_cmap(y, cmap)

        df["x"] = x
        df["y"] = y
        df["dot_size"] = df.y.map(size_func)

        _plot_dots(ax, x, y, df.dot_size.values, y, cmap, norm, highlight)

        if len(var_names) > 0:
            sub = df[df.gene.isin(var_names)]

            coords[i, :, 0] = sub.x.values
            coords[i, :, 1] = sub.y.values

            if i in show_labels:
                annotations = _annotate_dots(ax, sub, ["x", "y"], fontsize, False)
                texts.extend(annotations)

            if highlight:
                _ = _plot_dots(
                    ax,
                    sub.x.values,
                    sub.y.values,
                    sub.dot_size.values,
                    sub.y.values,
                    cmap,
                    norm,
                    False,
                )

    if len(var_names) > 0:
        for j in range(len(states) - 1):
            for g in range(coords.shape[1]):
                ax.plot(
                    [coords[j, g, 0], coords[j + 1, g, 0]],
                    [coords[j, g, 1], coords[j + 1, g, 1]],
                    alpha=connection_alpha,
                    color="k",
                    linestyle="--",
                    lw=connection_linewidth,
                )

    ax.set_xticks([i for i in range(len(states))])
    ax.set_title(f"Factor {factor}")
    ax.set_ylabel("Loading weight")
    ax.set_xlabel("State")
    ax.set_xticklabels(states)

    # style the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.grid()
    ax.set_axisbelow(True)

    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=annotation_linewidth), ax=ax)

    return ax
