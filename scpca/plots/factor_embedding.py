from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from scanpy.plotting._tools.scatterplots import _get_palette  # type: ignore

from ..utils.data import _validate_sign
from .helper import _set_up_cmap, _set_up_plot


def factor_embedding(
    adata: AnnData,
    model_key: str,
    factor: Union[int, List[int], None] = None,
    basis: Union[str, None] = None,
    sign: Union[float, int] = 1.0,
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    pad: float = 0.1,
    size: float = 1,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Axes = None,
) -> Axes:
    """
    Plot factor weights on a given basis such as UMAP/TSNE.

    Parameters
    ----------
    adata
        AnnData object.
    model_key
        Key for the fitted model.
    factor
        Factor(s) to plot. If None, then all factors are plotted.
    basis
        Key for the basis (e.g. UMAP, T-SNE). If basis is None factor embedding
        tries to retrieve "X_{model_key}_umap".
    sign
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap
        Colormap for the scatterplot.
    colorbar_pos
        Position of the colorbar.
    colorbar_width
        Width of the colorbar.
    orientation
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    pad
        Padding between the plot and colorbar
    size
        Marker/Dot size of the scatterplot.
    ncols
        Number of columns for the subplots.
    width
        Width of each subplot.
    height
        Height of each subplot.
    ax
        Axes object to plot on. If None, then a new figure is created. Works only
        if one factor is plotted.

    Returns
    -------
        Axes object.
    """
    # do validation here
    sign = _validate_sign(sign)

    if basis is None:
        basis = f"X_{model_key}_umap"

    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _factor_embedding,
        basis=basis,
        sign=sign,
        cmap=cmap,
        colorbar_pos=colorbar_pos,
        colorbar_width=colorbar_width,
        orientation=orientation,
        pad=pad,
        size=size,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _factor_embedding(
    adata: AnnData,
    model_key: str,
    factor: int,
    basis: Union[str, None],
    sign: Union[float, int],
    cmap: str,
    colorbar_pos: str,
    colorbar_width: str,
    orientation: str,
    pad: float,
    size: float,
    ax: Axes = None,
) -> Axes:
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.gcf()

    weights = sign * adata.obsm[f"X_{model_key}"][..., factor]
    cmap, norm = _set_up_cmap(weights, cmap)

    im = ax.scatter(
        adata.obsm[basis][:, 0],
        adata.obsm[basis][:, 1],
        s=size,
        c=weights,
        norm=norm,
        cmap=cmap,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=pad)
    fig.colorbar(im, cax=cax, orientation=orientation)
    ax.set_title(f"Factor {factor}")
    ax.set_xlabel(f"{basis}")
    ax.set_ylabel(f"{basis}")
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def factor_density(
    adata: AnnData,
    model_key: str,
    cluster_key: str,
    factor: Union[int, List[int], None] = None,
    groups: Union[str, List[str]] = [],
    sign: Union[float, int] = 1.0,
    fill: bool = True,
    lw: float = 0.5,
    legend: bool = True,
    ax: Axes = None,
    size: float = 1,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
) -> Axes:
    """
    Plot Factor Density for Clusters in Single-Cell Analysis.

    Parameters
    ----------
    adata
        Anndata object containing single-cell data.
    model_key
        Key for the fitted model.
    factor
        Index of the factor to visualize.
    cluster_key
        Key for the cluster annotations in adata.obs.
    groups
        Specific cluster groups to highlight in the plot. If empty, all clusters are used. (Default: [])
    fill
        Whether to fill the density plot. (Default: True)
    lw
        Line width for the density plot. (Default: 0.5)
    legend
        Whether to display the legend. (Default: True)
    ax
        Matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
        Matplotlib axes containing the factor density plot.

    Notes
    -----
    This function plots the density of a specific factor's weights across clusters in single-cell analysis.
    The density plot shows the distribution of the factor weights for each cluster group.
    """

    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _factor_density,
        cluster_key=cluster_key,
        groups=groups,
        legend=legend,
        sign=sign,
        lw=lw,
        fill=fill,
        ncols=ncols,
        width=width,
        height=height,
        ax=ax,
    )
    return ax


def _factor_density(
    adata: AnnData,
    model_key: str,
    factor: int,
    cluster_key: str,
    groups: Union[str, List[str]] = [],
    sign: Union[float, int] = 1.0,
    fill: bool = True,
    lw: float = 0.5,
    legend: bool = True,
    ax: Axes = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    if isinstance(groups, str):
        groups = [groups]

    df = pd.DataFrame(
        {
            cluster_key: [c if c in groups else "_" + c for c in adata.obs[cluster_key].values]
            if len(groups) > 0
            else adata.obs[cluster_key].values,
            "factor": sign * adata.obsm[f"X_{model_key}"][..., factor],
        }
    )

    color_key = f"{cluster_key}_colors"
    if color_key not in adata.uns:
        color_dict = _get_palette(adata, cluster_key)
        adata.uns[color_key] = [color_dict[k] for k in adata.obs[cluster_key].cat.categories]

    if len(groups) > 0:
        palette = {
            t if t in groups else "_" + t: c if t in groups else "lightgrey"
            for t, c in zip(adata.obs[cluster_key].cat.categories, adata.uns[color_key])
        }
    else:
        palette = {t: c for t, c in zip(adata.obs[cluster_key].cat.categories, adata.uns[color_key])}

    ax = sns.kdeplot(x="factor", hue=cluster_key, data=df, palette=palette, fill=fill, lw=lw, legend=legend, ax=ax)

    ax.set_xlabel("Factor weight")
    ax.set_title(f"Factor {factor}")
    ax.axvline(0, color="k", ls="--")

    return ax


def factor_strip(
    adata: AnnData,
    model_key: str,
    factor: Union[int, List[int]],
    cluster_key: str,
    highlight: Optional[Union[str, List[str]]] = None,
    state_key: Optional[Union[str]] = None,
    swap_axes: bool = False,
    sign: Union[int, float] = 1.0,
    kind: str = "strip",
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Axes = None,
    **kwargs: Any,
) -> Axes:
    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _factor_strip,
        cluster_key=cluster_key,
        highlight=highlight,
        state_key=state_key,
        sign=sign,
        kind=kind,
        swap_axes=swap_axes,
        sharex=True,
        width=width,
        height=height,
        ncols=ncols,
        ax=ax,
        **kwargs,
    )
    return ax


def _factor_strip(
    adata: AnnData,
    model_key: str,
    factor: int,
    cluster_key: str,
    state_key: Optional[str] = None,
    highlight: Optional[str] = None,
    sign: Union[float, int] = 1.0,
    kind: str = "strip",
    swap_axes: bool = False,
    ax: Axes = None,
    **kwargs: Any,
) -> Axes:
    _ = _validate_sign(sign)
    plot_funcs = {
        "strip": sns.stripplot,
        "box": sns.boxplot,
    }

    df = pd.DataFrame(sign * adata.obsm[f"X_{model_key}"]).assign(cluster=adata.obs[cluster_key].tolist())

    if highlight is not None:
        df = df.assign(highlight=lambda df: df.cluster.apply(lambda x: x if x in highlight else "other"))

    groupby_vars = ["cluster"]

    if state_key is not None:
        df[state_key] = adata.obs[state_key].tolist()
        groupby_vars.append(state_key)

    if state_key is None and highlight is not None:
        state_key = "highlight"
        groupby_vars.append(state_key)

    df = df.melt(groupby_vars, var_name="factor")
    # import pdb; pdb.set_trace()
    if swap_axes:
        g = plot_funcs[kind](y="cluster", x="value", hue=state_key, data=df[df["factor"] == factor], ax=ax, **kwargs)
        # g.axes.tick_params(axis="x", rotation=90)
        g.axes.axvline(0, color="k", linestyle="-", lw=0.5)
        g.axes.xaxis.grid(True)
        g.axes.set_xlabel("Factor weight")
    else:
        g = plot_funcs[kind](x="cluster", y="value", hue=state_key, data=df[df["factor"] == factor], ax=ax, **kwargs)
        g.axes.tick_params(axis="x", rotation=90)
        g.axes.axhline(0, color="k", linestyle="-", lw=0.5)
        g.axes.yaxis.grid(True)
        g.axes.set_ylabel("Factor weight")
    g.axes.spines["top"].set_visible(False)
    g.axes.spines["bottom"].set_visible(False)
    g.axes.spines["right"].set_visible(False)
    g.axes.spines["left"].set_visible(False)
    g.axes.set_title(f"Factor {factor}")

    return df
