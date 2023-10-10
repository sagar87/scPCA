from typing import Any, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from anndata import AnnData  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from numpy.typing import NDArray
from scipy.stats import gamma  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

from ..utils.data import _get_rna_counts


def _var(concentration: Union[float, NDArray[np.float32]], mean: Union[float, NDArray[np.float32]]) -> Any:
    """Computes the expected variance of the Gamma Poission distribution."""
    return concentration / (concentration / mean) ** 2 * (1 + concentration / mean)


def true_pred(
    adata: AnnData,
    model_key: str,
    layers_key: Optional[str] = None,
    cmap: str = "viridis",
    colorbar_pos: str = "right",
    colorbar_width: str = "3%",
    orientation: str = "vertical",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Create a 2D histogram plot of predicted vs. true counts of a scPCA model.

    Parameters
    ----------
    adata
        Anndata object containing single-cell RNA-seq data.
    model_key
        Key for the fitted model within the AnnData object.
    layers_key
        Key to extract counts from adata.layers. If None, raw counts are extracted from adata.X.
    cmap
        Colormap for the scatterplot. Default is "viridis".
    colorbar_pos
        Position of the colorbar (e.g., "right" or "left"). Default is "right".
    colorbar_width
        Width of the colorbar as a percentage of the figure. Default is "3%".
    orientation
        Orientation of the colorbar ("vertical" or "horizontal"). Default is "vertical".
    ax
        Existing matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
        Matplotlib axes containing the 2D scatter plot.

    Notes
    -----
    This function generates a 2D-histogram plot comparing predicted counts (from a fitted model)
    against true counts in single-cell RNA-seq data.
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


def mean_var(
    adata: AnnData,
    model_key: Optional[str] = None,
    layers_key: Optional[str] = None,
    β_rna_mean: float = 3,
    β_rna_sd: float = 1,
    alpha: float = 1.0,
    margin: float = 0.01,
    cmap: str = "viridis",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Create a Scatter Plot of Mean vs. Variance in Single-Cell RNA-seq Data.

    Parameters
    ----------
    adata
        Anndata object containing single-cell RNA-seq data.
    model_key
        Key for the fitted model within the AnnData object. (Default: None)
    layers_key
        Key to extract counts from adata.layers. If None, raw counts are extracted from adata.X.
    β_rna_mean
        Prior mean for RNA expression. (Default: 3.0)
    β_rna_sd
        Prior standard deviation for RNA expression. (Default: 1.0)
    alpha
        Transparency of data points in the scatter plot. (Default: 1.0)
    margin
        Margin for the variance plot. (Default: 0.01)
    cmap
        Colormap for the scatterplot. (Default: "viridis")
    ax
        Existing matplotlib axes to plot on. If None, a new figure is created.

    Returns
    -------
        Matplotlib axes containing the scatter plot.

    Notes
    -----
    This function creates a scatter plot of the mean expression values against the variance
    for genes in single-cell RNA-seq data. It also visualizes a prior distribution for variance.
    The plot may include labeled genes with the highest deviations from the expected variance.
    """

    if ax is None:
        plt.figure()
        ax = plt.gca()

    counts = _get_rna_counts(adata, layers_key)

    if model_key is not None:
        model_dict = adata.uns[model_key]
        params = model_dict["model"]
        # prior_mean = params["β_rna_mean"]
        β_rna_mean = params.get("β_rna_mean", 3.0)
        β_rna_sd = params.get("β_rna_sd", 0.01)

    a = β_rna_mean**2 / β_rna_sd**2
    b = β_rna_mean / β_rna_sd**2

    upper = gamma(a, scale=1 / b).ppf(0.975)
    lower = gamma(a, scale=1 / b).ppf(0.025)

    true_mean = np.mean(counts, axis=0)
    true_var = np.var(counts, axis=0)
    theoretical = _var(β_rna_mean, np.logspace(-4, 3, 1000))
    # expectation = _var(β_rna_mean, true_mean)

    ax.fill_between(
        np.logspace(-4, 3, 1000),
        _var(lower, np.logspace(-4, 3, 1000)),
        _var(upper, np.logspace(-4, 3, 1000)),
        color="C3",
        alpha=0.2,
    )
    im = ax.scatter(
        true_mean,
        true_var,
        alpha=alpha,
        s=10,
        c=1.0 / adata.varm[f"α_{model_key}"] if model_key is not None else None,
        cmap="viridis",
        norm=LogNorm(),
    )

    ax.plot(np.logspace(-4, 3), np.logspace(-4, 3), color="C3", label="Identity")
    ax.plot(
        np.logspace(-4, 3, 1000),
        theoretical,
        color="C3",
        linestyle="--",
        label=f"Prior mean {β_rna_mean:.2f}",
    )

    ax.legend()
    ax.set_ylabel("Variance")
    ax.set_xlabel("Mean")

    if model_key is not None:
        cax = plt.colorbar(im)
        cax.set_label("α")

    ax.set_yscale("log")
    ax.set_xscale("log")

    return ax
