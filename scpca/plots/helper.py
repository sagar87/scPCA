from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.colors as co  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from anndata import AnnData  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.cm import get_cmap  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from numpy.typing import NDArray


def set_up_cmap(
    array: NDArray[np.float32], colormap: str = "RdBu"
) -> Tuple[Union[co.Colormap, co.LinearSegmentedColormap], Union[co.TwoSlopeNorm, co.Normalize]]:
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


def rand_jitter(arr: NDArray[np.float32], stdev: float = 1.0) -> NDArray[np.float32]:
    # stdev = .01 * (max(arr) - min(arr))
    # print(stdev)
    return arr + np.random.randn(len(arr)) * stdev


def set_up_subplots(
    num_plots: int, ncols: int = 4, width: float = 4, height: float = 3, sharey: bool = False, sharex: bool = False
) -> Tuple[Figure, Axes]:
    """
    Set up subplots for plotting multiple factors.

    Parameters
    ----------
    num_plots : int
        The total number of plots to be created.
    ncols : int, optional
        The number of columns in the subplot grid. Default is 4.
    width : int, optional
        The width factor for each subplot. Default is 4.
    height : int, optional
        The height factor for each subplot. Default is 3.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object representing the entire figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
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
    fig, ax = set_up_subplots(num_plots=1)

    # Create a grid of subplots with 2 rows and 4 columns
    fig, axes = set_up_subplots(num_plots=8)

    # Create a single row of subplots with 1 row and 3 columns
    fig, axes = set_up_subplots(num_plots=3, ncols=3)
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


def set_up_plot(
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
    **kwargs: Any
) -> Axes:
    """
    Set up the plot environment for visualizing multiple instances or factors.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data and model information.
    model_key : str
        Key to access the model information in `adata.uns`.
    instances : int, List[int], or None
        Index or list of indices of instances or factors to visualize.
        If None, the function determines the number of instances/factors automatically.
    func : Callable
        Plotting function to visualize each instance/factor.
        It should accept the following parameters: `adata`, `model_key`, `instance`, and `ax`.
    ncols : int, optional (default: 4)
        Number of columns in the subplot grid.
    width : int, optional (default: 4)
        Width of each subplot in inches.
    height : int, optional (default: 3)
        Height of each subplot in inches.
    ax : plt.Axes or None, optional (default: None)
        Matplotlib axes to use for plotting. If None, new subplots will be created.
    **kwargs
        Additional keyword arguments to pass to the `func` plotting function.

    Returns
    -------
    ax : plt.Axes
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
    set_up_plot(adata, 'pca', 0, plot_function)

    # Plot multiple instances of a model using a custom plotting function
    set_up_plot(adata, 'umap', [0, 1, 2], plot_function)

    # Automatically determine the number of instances and plot them using a custom plotting function
    set_up_plot(adata, 'lda', None, plot_function)

    # Specify the number of columns and size of subplots
    set_up_plot(adata, 'nmf', [0, 1, 2, 3], plot_function, ncols=3, width=6, height=4)
    """
    if isinstance(instances, list):
        num_plots = len(instances)
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height, sharex=sharex, sharey=sharey)
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
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height, sharex=sharex, sharey=sharey)

    if num_plots == 1:
        if isinstance(instances, list):
            instances = instances[0]

        func(adata, model_key, instances, ax=ax, **kwargs)
    else:
        for i, ax_i in zip(instances, ax.flatten()):  # type: ignore
            func(adata, model_key, i, ax=ax_i, **kwargs)

    return ax
