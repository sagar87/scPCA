from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from adjustText import adjust_text  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

from ..logger import logger
from ..utils import state_diff
from ..utils.data import _validate_sign, _validate_states
from .helper import _annotate_dots, _plot_dots, _set_up_cmap, _set_up_plot


def loadings_state(
    adata: AnnData,
    model_key: str,
    states: Union[List[str], Tuple[str, str], str],
    factor: Union[int, List[int], None] = None,
    var_names: Union[List[str], str] = [],
    variable: str = "W",
    highest: int = 0,
    lowest: int = 0,
    sign: Union[int, float] = 1.0,
    highlight: bool = True,
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    fontsize: int = 10,
    pad: float = 0.1,
    show_corr: bool = False,
    show_diff: bool = False,
    show_orthants: bool = False,
    size_func: Callable[[float], float] = lambda x: 10.0,
    sharey: bool = False,
    sharex: bool = False,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    text_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the loadings of two states.

    Parameters
    ----------
    adata
        AnnData object containing the scPCA model.
    model_key
        Key to access the model information in `adata.uns`.
    states
        States to compare.
    factor
        Factor or list of factors to plot. If None, all factors are plotted. Default is None.
    var_names
        Variable names to highlight. Default is an empty list.
    variable
        Variable to plot. Default is "W".
    highest
        Number of genes to plot with largest positve difference between two states. Default is 0.
    lowest
        Number of genes to plot with largest negative difference between two states. Default is 0.
    sign
        Sign of the loadings, either -1 or 1. Default is 1.0.
    highlight
        If true only var_names and highest/lowest genes are shown in color, all other genes in grey.
        Default is True.
    cmap
        Colormap to use. Default is "RdBu".
    colorbar_pos
        Position of the colorbar. Default is "right".
    colorbar_width
        Width of the colorbar. Default is "3%".
    orientation
        Orientation of the colorbar. Default is "vertical".
    fontsize
        Font size for annotations. Default is 10.
    pad
        Padding for the colorbar. Default is 0.1.
    show_corr
        Whether to show correlation. Default is False.
    show_diff
        Whether to show log differences. Default is False.
    show_orthants
        Whether to show orthants.
    size_func
        Function to determine the size of the dots. Default is a lambda function that returns an array of 10.0.
    sharey
        Whether to share the y-axis across subplots. Default is False.
    sharex
        Whether to share the x-axis across subplots. Default is False.
    ncols
        Number of columns in the subplot grid. Default is 4.
    width
        Width of each subplot in inches. Default is 4.
    height
        Height of each subplot in inches. Default is 3.
    text_kwargs
        Additional keyword arguments for text annotations. Default is an empty dictionary.
    ax
        Matplotlib axes to use for plotting. If None, new subplots will be created. Default is None.

    Returns
    -------
    ax
        Matplotlib axes object containing the plotted instances or factors.

    Notes
    -----
    - The function sets up the plot environment and calls the internal function `_loadings_state` to do the actual plotting.
    - If `var_names` is provided, the arguments `highest` and `lowest` are ignored.

    Examples
    --------
    # Example usage
    loadings_state(adata, 'model1', ['state1', 'state2'], factor=0)
    """
    sign = _validate_sign(sign)
    states = _validate_states(states)

    if isinstance(var_names, str):
        var_names = [var_names]

    if len(var_names) > 0 and (highest > 0 or lowest > 0):
        logger.warning("If var_names are provided the arguments are highest and lowest are ignored.")

    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _loadings_state,
        states=states,
        var_names=var_names,
        variable=variable,
        highest=highest,
        lowest=lowest,
        sign=sign,
        highlight=highlight,
        size_func=size_func,
        colorbar_pos=colorbar_pos,
        colorbar_width=colorbar_width,
        orientation=orientation,
        fontsize=fontsize,
        pad=pad,
        cmap=cmap,
        show_corr=show_corr,
        show_diff=show_diff,
        show_orthants=show_orthants,
        sharey=sharey,
        sharex=sharex,
        ncols=ncols,
        width=width,
        height=height,
        text_kwargs=text_kwargs,
        ax=ax,
    )
    return ax


def _loadings_state(
    adata: AnnData,
    model_key: str,
    factor: int,
    states: List[str],
    var_names: List[str],
    variable: str,
    highest: int,
    lowest: int,
    sign: Union[float, int],
    size_func: Callable[[float], float],
    highlight: bool,
    cmap: str,
    colorbar_pos: str,
    colorbar_width: str,
    orientation: str,
    pad: float,
    fontsize: int,
    show_corr: bool,
    show_diff: bool,
    show_orthants: bool,
    text_kwargs: Dict[str, Any],
    ax: Axes = None,
) -> Axes:
    df = state_diff(adata, model_key, states, factor, variable=variable, highest=adata.shape[1], lowest=0, sign=sign)
    x = df[states[0]].values
    y = df[states[1]].values
    diff = df.difference.values
    cmap, norm = _set_up_cmap(diff, cmap)
    df["dot_size"] = df.difference.map(size_func)

    im = _plot_dots(ax, x, y, df.dot_size.values, diff, cmap, norm, highlight)

    # set up the diagonal line
    diag = np.linspace(np.quantile(df[states[0]].values, 0.01), np.quantile(df[states[0]].values, 0.99))
    ax.plot(diag, diag, ls="--", color="k", lw=0.5)

    # labels
    ax.set_xlabel(f"Loadings ({states[0]})")
    ax.set_ylabel(f"Loadings ({states[1]})")
    ax.set_title(f"Factor {factor}")

    texts = []

    if len(var_names) > 0:
        sub = df[df.gene.isin(var_names)]
        annotations = _annotate_dots(ax, sub, states, fontsize, show_diff)
        texts.extend(annotations)
        if highlight:
            im = _plot_dots(
                ax,
                sub[states[0]].values,
                sub[states[1]].values,
                sub.dot_size.values,
                sub.difference,
                cmap,
                norm,
                False,
            )

    elif (highest > 0) or (lowest > 0):
        sub = state_diff(adata, model_key, states, factor, variable=variable, highest=highest, lowest=lowest, sign=sign)
        sub["dot_size"] = sub.difference.map(size_func)
        annotations = _annotate_dots(ax, sub, states, fontsize, show_diff)
        texts.extend(annotations)
        if highlight:
            im = _plot_dots(
                ax,
                sub[states[0]].values,
                sub[states[1]].values,
                sub.dot_size.values,
                sub.difference,
                cmap,
                norm,
                False,
            )

    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), ax=ax, **text_kwargs)

    # set up color bar
    if (not highlight) or (highlight and (len(var_names) > 0 or ((highest > 0) or (lowest > 0)))):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=pad)
        plt.gcf().colorbar(im, cax=cax, orientation=orientation)

    if show_corr:
        correlation = np.corrcoef(x, y)[0, 1]
        ax.text(0.95, 0.95, f"Correlation: {correlation:.2f}", ha="right", va="top", transform=ax.transAxes)

    if show_orthants:
        ax.axvline(0, ls="--", color="k", lw=0.5)
        ax.axhline(0, ls="--", color="k", lw=0.5)

    return ax


def loading_rank_diff(
    adata: AnnData,
    model_key: str,
    states: Union[List[str], Tuple[str, str], str],
    factor: Union[int, List[int]],
    fontsize: int = 8,
    width: float = 1.0,
    height: float = 5,
    highest: int = 0,
    lowest: int = 0,
    magnitude: Optional[float] = None,
) -> Axes:
    num_genes = adata.shape[1]
    if isinstance(factor, int):
        factor = [factor]

    fig = plt.figure(figsize=(len(factor) * width, height))
    gs = fig.add_gridspec(1, len(factor), hspace=0, wspace=0)
    axes = gs.subplots(sharex="col", sharey="row")

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])

    for i, f in enumerate(factor):
        df = state_diff(adata, model_key, states, f, highest=num_genes).assign(idx=np.arange(num_genes)[::-1])

        axes[i].scatter(df.idx.values, df["difference"].values, s=1)

        texts = []

        if magnitude is None and lowest > 0 and highest > 0:
            for idx, row in df.head(highest).iterrows():
                text = axes[i].text(row["idx"], row["difference"], row["gene"], fontsize=fontsize)
                texts.append(text)
            for idx, row in df.tail(lowest).iterrows():
                text = axes[i].text(row["idx"], row["difference"], row["gene"], fontsize=fontsize)
                texts.append(text)
        elif magnitude is None and highest > 0:
            for idx, row in df.head(highest).iterrows():
                text = axes[i].text(row["idx"], row["difference"], row["gene"], fontsize=fontsize)
                texts.append(text)
        elif magnitude is None and lowest > 0:
            for idx, row in df.tail(lowest).iterrows():
                text = axes[i].text(row["idx"], row["difference"], row["gene"], fontsize=fontsize)
                texts.append(text)
        elif magnitude:
            for idx, row in df[df.magnitude > magnitude].iterrows():
                text = axes[i].text(row["idx"], row["difference"], row["gene"], fontsize=fontsize)
                texts.append(text)

        if len(texts) > 0:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), ax=axes[i])
        axes[i].margins(x=0.1)
        axes[i].set_xticks([])
        axes[i].set_title(f"F{f}")

        if i % 2 == 0:
            axes[i].set_facecolor("lightgrey")

    axes[0].set_ylabel("Difference")

    return axes
