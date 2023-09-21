from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from adjustText import adjust_text  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import Colormap, Normalize  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from numpy.typing import NDArray

from ..logger import logger
from ..utils import state_diff
from ..utils.data import _validate_sign, _validate_states
from .helper import _set_up_cmap, _set_up_plot


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
    size_func: Callable[[NDArray[np.float32]], NDArray[np.float32]] = lambda x: np.asarray([10.0]),
    sharey: bool = False,
    sharex: bool = False,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    text_kwargs: Dict[str, Any] = {},
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the loading states for a given model and factors.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the scPCA model.
    model_key : str
        Key to access the model information in `adata.uns`.
    states : Union[List[str], Tuple[str, str], str]
        States to compare.
    factor : Union[int, List[int], None], optional
        Factor or list of factors to plot. If None, all factors are plotted. Default is None.
    var_names : Union[List[str], str], optional
        Variable names to highlight. Default is an empty list.
    variable : str, optional
        Variable to plot. Default is "W".
    highest : int, optional
        Number of genes to plot with largest positve difference between two states. Default is 0.
    lowest : int, optional
        Number of genes to plot with largest negative difference between two states. Default is 0.
    sign : Union[int, float], optional
        Sign of the loadings, either -1 or 1. Default is 1.0.
    highlight : bool, optional
        If true only var_names and highest/lowest genes are shown in color, all other genes in grey.
        Default is True.
    cmap : str, optional
        Colormap to use. Default is "RdBu".
    colorbar_pos : str, optional
        Position of the colorbar. Default is "right".
    colorbar_width : str, optional
        Width of the colorbar. Default is "3%".
    orientation : str, optional
        Orientation of the colorbar. Default is "vertical".
    fontsize : int, optional
        Font size for annotations. Default is 10.
    pad : float, optional
        Padding for the colorbar. Default is 0.1.
    show_corr : bool, optional
        Whether to show correlation. Default is False.
    show_diff : bool, optional
        Whether to show difference. Default is False.
    size_func : Callable, optional
        Function to determine the size of the dots. Default is a lambda function that returns an array of 10.0.
    sharey : bool, optional
        Whether to share the y-axis across subplots. Default is False.
    sharex : bool, optional
        Whether to share the x-axis across subplots. Default is False.
    ncols : int, optional
        Number of columns in the subplot grid. Default is 4.
    width : int, optional
        Width of each subplot in inches. Default is 4.
    height : int, optional
        Height of each subplot in inches. Default is 3.
    text_kwargs : Dict[str, Any], optional
        Additional keyword arguments for text annotations. Default is an empty dictionary.
    ax : Optional[Axes], optional
        Matplotlib axes to use for plotting. If None, new subplots will be created. Default is None.

    Returns
    -------
    ax : plt.Axes
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
    var_names: List[str] = [],
    variable: str = "W",
    highest: int = 0,
    lowest: int = 0,
    sign: Union[float, int] = 1.0,
    size_func: Callable[[NDArray[np.float32]], NDArray[np.float32]] = lambda x: np.asarray([10.0]),
    highlight: bool = False,
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    pad: float = 0.1,
    fontsize: int = 10,
    show_corr: bool = False,
    show_diff: bool = False,
    text_kwargs: Dict[str, Any] = {},
    ax: Axes = None,
) -> Axes:
    df = state_diff(adata, model_key, states, factor, variable=variable, highest=adata.shape[1], lowest=0, sign=sign)
    x = df[states[0]].values
    y = df[states[1]].values
    diff = df.difference.values
    cmap, norm = _set_up_cmap(diff, cmap)
    size = size_func(diff)

    im = _plot_dots(ax, x, y, size, diff, cmap, norm, highlight)

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
                size_func(sub.difference.values),
                sub.difference,
                cmap,
                norm,
                False,
            )

    elif (highest > 0) or (lowest > 0):
        sub = state_diff(adata, model_key, states, factor, variable=variable, highest=highest, lowest=lowest, sign=sign)
        annotations = _annotate_dots(ax, sub, states, fontsize, show_diff)
        texts.extend(annotations)
        if highlight:
            im = _plot_dots(
                ax,
                sub[states[0]].values,
                sub[states[1]].values,
                size_func(sub.difference.values),
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

    return ax


def _plot_dots(
    ax: Axes,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    size: NDArray[np.float32],
    diff: NDArray[np.float32],
    cmap: Colormap,
    norm: Normalize,
    highlight: bool,
) -> Any:
    if highlight:
        im = ax.scatter(x, y, s=size, c="lightgrey")
    else:
        im = ax.scatter(x, y, s=size, c=diff, cmap=cmap, norm=norm)

    return im


def _annotate_dots(
    ax: Axes, dataframe: pd.DataFrame, states: List[str], fontsize: int = 10, show_diff: bool = False
) -> List[Any]:
    texts = []
    for i, row in dataframe.iterrows():
        # import pdb; pdb.set_trace()
        label = row["gene"]

        if show_diff:
            label += f' {row["difference"]:.2f}'

        t = ax.text(row[states[0]], row[states[1]], s=label, fontsize=fontsize)

        texts.append(t)

    return texts
