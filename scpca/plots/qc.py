from typing import Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from anndata import AnnData  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore

# from scipy.stats import gamma, variation  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

from ..utils.data import _get_rna_counts


def qc_hist(
    adata: AnnData,
    model_key: str,
    layers_key: Union[str, None] = None,
    cmap: str = "viridis",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plots a 2D histogram of the predicted counts against the true counts.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    model_key: str, optional (default: "scpca")
        Key for the fitted model.
    layers_key: str, optional (default: None)
        If `layers_key` is None, then the raw counts are extracted from `adata.X`.
        Otherwise, the counts are extracted from `adata.layers[layers_key]`.
    protein_obsm_key: str, optional (default: None)
        Key for protein counts in `adata.obsm`. Providing `protein_obsm_key`
        overrides `layers_key`, i.e. protein counts are plotted.
    cmap: str, optional (default: "viridis")
        Colormap for the scatterplot. Color represents the mean of the counts.
    ax: matplotlib.axes.Axes, optional (default: None)
        Axes to plot on. If None, then a new figure is created.

    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.gcf()

    # Extract counts
    counts = _get_rna_counts(adata, layers_key)
    predicted_counts = adata.layers[f"μ_{model_key}"]

    im = ax.hist2d(
        np.log10(counts.reshape(-1) + 1),
        np.log10(predicted_counts.reshape(-1) + 1),
        bins=50,
        norm=LogNorm(),
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=0.1)
    fig.colorbar(im[3], cax=cax, orientation=orientation)

    max_val = np.max([*ax.get_xlim(), *ax.get_ylim()])
    min_val = np.min([*ax.get_xlim(), *ax.get_ylim()])
    # print(max_val)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.plot(
        np.linspace(min_val, max_val),
        np.linspace(min_val, max_val),
        color="w",
        linewidth=2,
    )
    ax.set_aspect("equal")
    ax.set_ylabel(r"Predicted count ($\log_{10}(x+1)$ scaled)")
    ax.set_xlabel(r"True count ($\log_{10}(x+1)$ scaled)")
    rmse = mean_squared_error(counts, predicted_counts)
    ax.set_title(f"RMSE {rmse:.2f}")
    return ax
