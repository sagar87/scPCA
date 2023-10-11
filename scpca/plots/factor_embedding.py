from typing import List, Union

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
    ax
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
    fill: bool = True,
    lw: float = 0.5,
    legend: bool = True,
    ax: Axes = None,
    size: float = 1,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
) -> Axes:
    # do validation here

    ax = _set_up_plot(
        adata,
        model_key,
        factor,
        _factor_density,
        cluster_key=cluster_key,
        groups=groups,
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
            "factor": adata.obsm[f"X_{model_key}"][..., factor],
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
