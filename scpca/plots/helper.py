from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.colors as co  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.cm import get_cmap  # type: ignore
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure  # type: ignore
from numpy.typing import NDArray


def _set_up_cmap(
    array: NDArray[np.float32], colormap: str = "RdBu"
) -> Tuple[Union[co.Colormap, co.LinearSegmentedColormap], Union[co.TwoSlopeNorm, co.Normalize]]:
    """
    Set up a colormap and normalization based on the given array.

    Parameters
    ----------
    array :
        Input array for which the colormap and normalization are to be set up.
    colormap :
        The name of the colormap to use. Default is "RdBu".

    Returns
    -------
        A tuple containing the colormap and normalization objects.

    Notes
    -----
    - If the array contains both positive and negative values, a diverging colormap is used.
    - If the array contains only negative values, a colormap ranging from the minimum value to zero is used.
    - If the array contains only non-negative values, a colormap ranging from zero to the maximum value is used.

    Examples
    --------
    >>> cmap, norm = _set_up_cmap(np.array([-1, 0, 1]))
    >>> cmap, norm = _set_up_cmap(np.array([-1, -0.5, -0.2]), colormap="coolwarm")
    """

    vmin = array.min()
    vmax = array.max()

    if vmin < 0 and vmax > 0:
        norm = co.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        cmap = get_cmap(colormap)
    elif vmin < 0 and vmax < 0:
        # print('min color')
        norm = co.Normalize(vmin=vmin, vmax=0)
        cmap = co.LinearSegmentedColormap.from_list("name", [cmap(-0.001), "w"])
    else:
        # print('max color')
        cmap = co.LinearSegmentedColormap.from_list("name", ["w", cmap(1.001)])
        norm = co.Normalize(vmin=0, vmax=vmax)

    return cmap, norm


def _rand_jitter(arr: NDArray[np.float32], stdev: float = 1.0) -> NDArray[np.float32]:
    """
    Add random jitter to an array.

    Parameters
    ----------
    arr :
        Input array to which random jitter will be added.
    stdev :
        Standard deviation of the random jitter. Default is 1.0.

    Returns
    -------
        Array with added random jitter.

    Examples
    --------
    >>> jittered_arr = _rand_jitter(np.array([1, 2, 3]))
    >>> jittered_arr = _rand_jitter(np.array([1, 2, 3]), stdev=0.5)
    """
    return arr + np.random.randn(len(arr)) * stdev


def _set_up_subplots(
    num_plots: int, ncols: int = 4, width: float = 4, height: float = 3, sharey: bool = False, sharex: bool = False
) -> Tuple[Figure, Axes]:
    """
    Internal function to set up subplots for plotting multiple factors.

    Parameters
    ----------
    num_plots :
        The total number of plots to be created.
    ncols :
        The number of columns in the subplot grid. Default is 4.
    width :
        The width factor for each subplot. Default is 4.
    height :
        The height factor for each subplot. Default is 3.
    sharey :
        Whether to share the y-axis across subplots. Default is False.
    sharex :
        Whether to share the x-axis across subplots. Default is False.

    Returns
    -------
        The Figure object representing the entire figure.
        An array of Axes objects representing the subplots. The shape of the
        array is determined by the number of rows and columns.

    Notes
    -----
    - If `num_plots` is 1, a single subplot is created and returned as `fig` and `ax`.
    - If `num_plots` is less than `ncols`, a single row of subplots is created.
    - If `num_plots` is greater than or equal to `ncols`, a grid of subplots is created.
    - The `axes` array may contain empty subplots if the number of plots is less than the total available subplots.

    Examples
    --------
    # Create a single subplot
    fig, ax = _set_up_subplots(num_plots=1)

    # Create a grid of subplots with 2 rows and 4 columns
    fig, axes = _set_up_subplots(num_plots=8)

    # Create a single row of subplots with 1 row and 3 columns
    fig, axes = _set_up_subplots(num_plots=3, ncols=3)
    """

    if num_plots == 1:
        fig, ax = plt.subplots()
        return fig, ax

    nrows, reminder = divmod(num_plots, ncols)

    if num_plots < ncols:
        nrows = 1
        ncols = num_plots
    else:
        nrows, reminder = divmod(num_plots, ncols)

        if nrows == 0:
            nrows = 1
        if reminder > 0:
            nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), sharey=sharey, sharex=sharex)
    _ = [ax.axis("off") for ax in axes.flatten()[num_plots:]]
    return fig, axes


def _set_up_plot(
    adata: AnnData,
    model_key: str,
    instances: Union[int, List[int], None],
    func: Callable[..., None],
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    sharey: bool = False,
    sharex: bool = False,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """
    Set up the plot environment for visualizing multiple instances or factors.

    Parameters
    ----------
    adata :
        AnnData object containing the data and model information.
    model_key :
        Key to access the model information in `adata.uns`.
    instances :
        Index or list of indices of instances or factors to visualize.
        If None, the function determines the number of instances/factors automatically.
    func :
        Plotting function to visualize each instance/factor.
        It should accept the following parameters: `adata`, `model_key`, `instance`, and `ax`.
    ncols :
        Number of columns in the subplot grid.
    width :
        Width of each subplot in inches.
    height :
        Height of each subplot in inches.
    ax :
        Matplotlib axes to use for plotting. If None, new subplots will be created.
    **kwargs
        Additional keyword arguments to pass to the `func` plotting function.

    Returns
    -------
        Matplotlib axes object containing the plotted instances or factors.

    Notes
    -----
    - If `instances` is an integer, only a single instance will be plotted.
    - If `instances` is a list of integers, each specified instance will be plotted in separate subplots.
    - If `instances` is None, the function will determine the number of instances automatically based on the `model_key`.
    - The `func` plotting function should accept the `adata`, `model_key`, `instance`, and `ax` parameters.
      It is responsible for plotting the specific instance or factor.

    Examples
    --------
    # Plot a single instance of a model using a custom plotting function
    _set_up_plot(adata, 'pca', 0, plot_function)

    # Plot multiple instances of a model using a custom plotting function
    _set_up_plot(adata, 'umap', [0, 1, 2], plot_function)

    # Automatically determine the number of instances and plot them using a custom plotting function
    _set_up_plot(adata, 'lda', None, plot_function)

    # Specify the number of columns and size of subplots
    _set_up_plot(adata, 'nmf', [0, 1, 2, 3], plot_function, ncols=3, width=6, height=4)
    """
    if isinstance(instances, list):
        num_plots = len(instances)
        fig, ax = _set_up_subplots(num_plots, ncols=ncols, width=width, height=height, sharex=sharex, sharey=sharey)
    elif isinstance(instances, int):
        num_plots = 1
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(width, height))
    else:
        model_dict = adata.uns[model_key]
        if model_key == "pca":
            num_plots = model_dict["variance"].shape[0]
        else:
            num_plots = model_dict["model"]["num_factors"]

        instances = [i for i in range(num_plots)]
        fig, ax = _set_up_subplots(num_plots, ncols=ncols, width=width, height=height, sharex=sharex, sharey=sharey)

    if num_plots == 1:
        if isinstance(instances, list):
            instances = instances[0]

        func(adata, model_key, instances, ax=ax, **kwargs)
    else:
        for i, ax_i in zip(instances, ax.flatten()):  # type: ignore
            func(adata, model_key, i, ax=ax_i, **kwargs)

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
