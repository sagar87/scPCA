from typing import List, Union

import matplotlib.pyplot as plt  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

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
    Plot factor on a given basis.

    Parameters
    ----------
    adata :
        AnnData object.
    model_key :
        Key for the fitted model.
    factor :
        Factor(s) to plot. If None, then all factors are plotted.
    basis :
        Key for the basis (e.g. UMAP, T-SNE). If basis is None factor embedding
        tries to retrieve "X_{model_key}_umap".
    sign :
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap :
        Colormap for the scatterplot.
    colorbar_pos :
        Position of the colorbar.
    colorbar_width :
        Width of the colorbar.
    orientation :
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    pad :
        Padding between the plot and colorbar
    size :
        Marker/Dot size of the scatterplot.
    ncols :
        Number of columns for the subplots.
    width :
        Width of each subplot.
    height :
        Height of each subplot.
    ax :
        Axes object to plot on. If None, then a new figure is created. Works only
        if one factor is plotted.

    Returns
    -------
    ax :
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
        sign=sign,
        cmap=cmap,
        basis=basis,
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
    basis: Union[str, None] = None,
    sign: Union[float, int] = 1.0,
    cmap: str = "RdBu",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    pad: float = 0.1,
    size: float = 1,
    ax: Axes = None,
) -> Axes:
    """
    Helper function to plot factor embeddings.

    Parameters
    ----------
    adata :
        AnnData object.
    model_key :
        Key for the fitted model.
    factor :
        Factor to plot.
    basis :
        Key for the basis (e.g. UMAP, T-SNE).
    sign :
        Sign of the factor. Should be either 1.0 or -1.0.
    cmap :
        Colormap for the scatterplot.
    colorbar_pos :
        Position of the colorbar.
    colorbar_width :
        Width of the colorbar.
    orientation :
        Orientation of the colorbar. Should be either "vertical" or "horizontal".
    pad :
        Padding for the colorbar.
    size :
        Marker/Dot size of the scatterplot.
    ax :
        Axes object to plot on. If None, then a new figure is created.

    Returns
    -------
    ax :
        Axes object.
    """

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
