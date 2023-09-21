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
    highlight: bool = False,
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    fontsize: int = 10,
    pad: float = 0.1,
    show_corr: bool = False,
    show_diff: bool = False,
    show_lines: bool = False,
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
    Visualize the loading state of a given model on the data.

    Parameters
    ----------
    adata :
        The annotated data matrix.
    model_key :
        Key to access the model from the data.
    states :
        List of states to consider. Default is an empty list.
    factor :
        Factor or factors to consider. Default is None.
    variable :
        Variable to consider. Default is "W_rna".
    highest :
        Number of highest values to consider. Default is 10.
    lowest :
        Number of lowest values to consider. Default is 10.
    threshold :
        Threshold value to consider. Default is None.
    sign :
        Sign value. Default is 1.0.
    geneset :
        Set of genes to consider. Default is None.
    geneset_top_genes :
        Number of top genes in the gene set to consider. Default is 100.
    geneset_bottom_genes :
        Number of bottom genes in the gene set to consider. Default is 0.
    organism :
        Organism to consider. Default is "Human".
    cmap
        Colormap to use. Default is cm.RdBu.
    colorbar_pos :
        Position of the colorbar. Default is "right".
    colorbar_width :
        Width of the colorbar. Default is "3%".
    orientation :
        Orientation of the plot. Default is "vertical".
    fontsize :
        Font size to use. Default is 10.
    pad :
        Padding value. Default is 0.1.
    show_corr :
        Whether to show correlation. Default is False.
    show_rank :
        Whether to show rank. Default is False.
    show_diff :
        Whether to show difference. Default is False.
    show_lines :
        Whether to show lines. Default is False.
    size_func :
        Function to determine size. Default is a function that returns 10.
    text_func :
        Function to format text. Default is a function that wraps text.
    sharey :
        Whether to share y-axis. Default is False.
    sharex :
        Whether to share x-axis. Default is False.
    ncols :
        Number of columns. Default is 4.
    width :
        Width of the plot. Default is 4.
    height :
        Height of the plot. Default is 3.
    text_kwargs :
        Keyword arguments for text formatting. Default is an empty dictionary.
    ax :
        Axis object. Default is None.

    Returns
    -------
    ax :
        Configured axis object.

    Notes
    -----
    This function visualizes the loading state of a given model on the data.
    It provides various customization options for visualization.
    """
    sign = _validate_sign(sign)
    states = _validate_states(states)

    if isinstance(var_names, str):
        var_names = [var_names]

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
        show_lines=show_lines,
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
    show_lines: bool = False,
    show_diff: bool = False,
    text_kwargs: Dict[str, Any] = {},
    ax: Axes = None,
) -> Axes:
    # state_a, state_b = model_design[states[0]], model_design[states[1]]
    # loadings = adata.varm[f"{variable}_{model_key}"]
    df = state_diff(adata, model_key, states, factor, variable=variable, highest=adata.shape[1], lowest=0, sign=sign)

    # x, y = loadings[..., factor, state_a], loadings[..., factor, state_b]
    # diff = sign * (y - x)
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
        # import pdb;pdb.set_trace()
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
            # im = _plot_dots(ax, sub, states, cmap, size_func, False)
        # im = ax.scatter(x, y, s=size, c=diff, cmap=cmap, norm=norm)

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

    # import pdb; pdb.set_trace()

    # if show_lines:
    #     ax.axvline(0, ls="--", color="k", lw=0.5)
    #     ax.axhline(0, ls="--", color="k", lw=0.5)
    # # ax.spines["top"].set_visible(False)
    # # ax.spines["right"].set_visible(False)
    # ax.set_aspect("equal")

    # texts = []
    # if geneset is not None:
    #     diff_geneset = get_diff_enrichment(
    #         adata,
    #         model_key,
    #         states,
    #         factor=factor,
    #         geneset=geneset,
    #         sign=sign,
    #         organism=organism,
    #         highest=geneset_top_genes,
    #         lowest=geneset_bottom_genes,
    #     )
    #     # import pdb;pdb.set_trace()

    #     for i, row in diff_geneset.head(highest).iterrows():
    #         label = str(row["Term"])

    #         if show_rank:
    #             label = f"{i+1} " + label
    #         genes = row["Genes"].split(";")
    #         # print(genes)
    #         # print(adata.var_names.isin(genes))
    #         # import pdb; pdb.set_trace()
    #         is_upper = np.all([gene.isupper() for gene in genes])

    #         var_names = adata.var_names
    #         if is_upper:
    #             var_names = var_names.str.upper()

    #         # gene_rep = np.random.choice(adata.var_names[adata.var_names.str.upper().isin(genes)])
    #         # import pdb;pdb.set_trace()
    #         # t = ax.text(x[adata.var_names == gene_rep].item(), y[adata.var_names == gene_rep].item(), s=text_func(label), fontsize=fontsize)
    #         t = ax.text(
    #             sign * x[var_names.isin(genes)].mean(),
    #             sign * y[var_names.isin(genes)].mean(),
    #             s=text_func(label),
    #             fontsize=fontsize,
    #         )
    #         texts.append(t)

    # else:
    #     if threshold:
    #         diff_genes = get_diff_genes(
    #             adata,
    #             model_key,
    #             states,
    #             factor,
    #             vector=variable,
    #             sign=sign,
    #             highest=adata.shape[1],
    #             threshold=threshold,
    #         )
    #         diff_genes = diff_genes[diff_genes.significant]
    #     else:
    #         diff_genes = get_diff_genes(
    #             adata, model_key, states, factor, vector=variable, sign=sign, highest=highest, lowest=lowest
    #         )

    # if show_corr:
    #     correlation = np.corrcoef(x, y)[0, 1]
    #     ax.text(0.95, 0.95, f"Correlation: {correlation:.2f}", ha="right", va="top", transform=ax.transAxes)

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
