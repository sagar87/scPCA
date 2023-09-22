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
    cmap: str = "RdBu",
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Axes = None,
) -> Axes:
    """
    Plot factor on a given embedding.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    model_key: str, optional (default: "X_scpca")
        Key for the fitted model.
    embedding: str, optional (default: "X_umap")
        Key for the embedding (e.g. UMAP, T-SNE).
    factor: int, list, optional (default: None)
        Factor(s) to plot. If None, then all factors are plotted.
    sign: float, optional (default: 1.0)
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap: str, optional (default: "PiYG")
        Colormap for the scatterplot.
    colorbar_pos: str, optional (default: "right")
        Position of the colorbar.
    colorbar_width: str, optional (default: "3%")
        Width of the colorbar.
    orientation: str, optional (default: "vertical")
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    size: float, optional (default: 1)
        Marker/Dot size of the scatterplot.
    ncols: int, optional (default: 4)
        Number of columns for the subplots.
    width: int, optional (default: 4)
        Width of each subplot.
    height: int, optional (default: 3)
        Height of each subplot.
    ax: matplotlib.axes.Axes, optional (default: None)
        Axes object to plot on. If None, then a new figure is created.

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes object.
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
    cmap: str,
    ax: Axes = None,
) -> Axes:
    """
    Scatter plot of factor loadings for a given factor in each state.

    Arguments
    ---------
    adata: AnnData
        AnnData object with the fitted model.
    model_key: str
        Key used to store the fitted model in adata.uns.
    factor: int
        The factor to plot.
    states: List[str], optional (default: [])
        The states to include in the plot.
    genes: List[str], optional (default: [])
        The genes to include in the plot.
    diff: List[str], optional (default: [])
        The genes to highlight in the plot.
    geneset: str or None, optional (default: None)
        Name of a gene set to include in the plot. Requires gseapy package.
    vector: str, optional (default: "W_rna")
        Vector to use for plotting the loadings.
    alpha: float, optional (default: 1.0)
        Transparency of the scatter plot.
    highest: int, optional (default: 3)
        Number of genes with highest loadings to plot per state.
    lowest: int, optional (default: 3)
        Number of genes with lowest loadings to plot per state.
    size_scale: float, optional (default: 1.0)
        Scaling factor for the gene symbol size.
    sign: float, optional (default: 1.0)
        Sign of the loadings.
    jitter: float, optional (default: 0.01)
        Jittering factor for the x-axis to reduce overlap.
    fontsize: int, optional (default: 10)
        Font size for gene labels.
    geneset_top_genes: int, optional (default: 100)
        Number of genes from the gene set to plot with the highest loadings.
    geneset_bottom_genes: int, optional (default: 0)
        Number of genes from the gene set to plot with the lowest loadings.
    show_labels: int, optional (default: 0)
        Show gene labels for top `show_labels` genes with the highest loadings.
    show_geneset: bool, optional (default: False)
        Show the gene set as a solid line.
    show_diff: bool, optional (default: False)
        Show the differential genes as a dashed line.
    return_order: bool, optional (default: False)
        Return the order of genes plotted.
    annotation_kwargs: dict, optional (default: {})
        Additional keyword arguments for gene label annotations.

    Returns
    -------
    order: np.ndarray
        The order of genes plotted (only if `return_order` is True).
    """
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
                    alpha=0.5,
                    color="k",
                    linestyle="--",
                    lw=0.1,
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
