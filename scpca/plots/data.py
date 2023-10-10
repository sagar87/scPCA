from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.collections as collections  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.colors import (  # type: ignore
    LinearSegmentedColormap,
    hsv_to_rgb,
    rgb_to_hsv,
)
from numpy.typing import NDArray


def _get_hsvcmap(i: int, N: int, rot: float = 0.0) -> LinearSegmentedColormap:
    nsc = 24
    chsv = rgb_to_hsv(plt.cm.hsv(((np.arange(N) / N) + rot) % 1.0)[i, :3])
    rhsv = rgb_to_hsv(plt.cm.Reds(np.linspace(0.2, 1, nsc))[:, :3])
    arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
    arhsv[:, 1:] = rhsv[:, 1:]
    rgb = hsv_to_rgb(arhsv)
    return LinearSegmentedColormap.from_list("", rgb)


def _triatpos(pos: Tuple[Union[float, int], Union[float, int]] = (0, 0), rot: float = 0) -> NDArray[np.float32]:
    r = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]]) * 0.5
    rm = [
        [np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
        [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))],
    ]
    r = np.dot(rm, r.T).T
    r[:, 0] += pos[0]
    r[:, 1] += pos[1]
    return r


def triangle_overlay(
    array: Optional[NDArray[np.float32]] = None,
    num_rows: int = 16,
    num_cols: int = 4,
    rot: int = 0,
    num_states: int = 2,
    cmap: List[str] = ["Blues", "Greens"],
    color: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> plt.Axes:
    # segs = []
    if ax is None:
        ax = plt.gca()

    size = (num_rows, num_cols)
    if array is None:
        array = np.random.normal(size=size).astype(np.float32)

    val_mapping = {i: r for i, r in enumerate(np.split(np.arange(num_rows), num_states))}
    row_mapping = {j: i for i, r in enumerate(np.split(np.arange(num_rows), num_states)) for j in r}
    segs_dict: Dict[int, List[NDArray[np.float32]]] = {i: [] for i in range(num_states)}

    for i in range(num_rows):
        for j in range(num_cols):
            k = row_mapping[i]
            segs_dict[k].append(_triatpos((j, i), rot=rot))

    for k, v in segs_dict.items():
        if color:
            col = collections.PolyCollection(v, color=color[k], **kwargs)
        else:
            col = collections.PolyCollection(v, cmap=cmap[k], color="w", **kwargs)
        if array is not None:
            col.set_array(array.flatten()[val_mapping[k]])
        ax.add_collection(col)

    return ax


def data_matrix(
    array: Optional[NDArray[np.float32]] = None,
    num_rows: int = 16,
    num_cols: int = 4,
    num_states: Union[int, List[int]] = 2,
    right_add: int = 0,
    cmaps: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    remove_ticklabels: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylabel_pos: Optional[Tuple[float, float]] = None,
    xlabel_pos: Optional[Tuple[float, float]] = None,
    hlinewidth: Optional[float] = None,
    vlinewidth: Optional[float] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Visualizes a data matrix with multiple colormaps.


    Parameters
    ----------
    size : tuple
        Size of the data matrix. If array is not None, size is ignored.
    array : np.ndarray
        Data matrix. If None, a random matrix is generated.
    right_add : int
        Number of columns to padd to the right of the data matrix. May be
        useful to add annotations on the right side of the matrix.
    num_states : int or list
        Number of categories to split the data matrix into. Must divide the
        array.shape[0] evenly.
    cmaps : list
        List of colormaps to use. If None, a default colormap is used.
    """
    # ax = ax or plt.gca()
    if ax is None:
        plt.figure(figsize=(0.8, 3))
        ax = plt.gca()

    rows = num_rows
    cols = num_cols
    size = (rows, cols)
    if array is None:
        array = np.random.normal(size=size).astype(np.float32)

    index_array = []

    if isinstance(num_states, int):
        sub_rows = int(rows / num_states)

        for i in range(num_states):
            index_array.append(i * np.ones((sub_rows, cols)))
    else:
        sub_rows = len(num_states)

        for i, j in enumerate(num_states):
            index_array.append(i * np.ones((j, cols)))

    premask = np.concatenate(index_array, 0)

    if right_add > 0:
        premask_pad = -np.ones((size[0], right_add))
        premask = np.concatenate([premask, premask_pad], 1)

        right_pad = np.zeros((size[0], right_add))
        array = np.concatenate([array, right_pad], 1)

    images = []

    if cmaps is None:
        cmap = [_get_hsvcmap(i, int(np.max(premask)) + 1, rot=0.5) for i in range(int(np.max(premask) + 1))]
    else:
        cmap = []

        for c in cmaps:
            try:
                cm = plt.get_cmap(c)
            except ValueError:
                cm = LinearSegmentedColormap.from_list(f"{c}", colors=["w", c])

            cmap.append(cm)

    for i in range(int(np.min(premask)), int(np.max(premask) + 1)):
        if i == -1:
            continue
        else:
            col = np.ma.array(array, mask=premask != i)  # type: ignore
            im = ax.imshow(col, cmap=cmap[i], vmin=vmin, vmax=vmax)
            images.append(im)

    xgrid = np.arange(size[0] + 1) - 0.5
    ygrid = np.arange(size[1] + 1) - 0.5
    ax.hlines(xgrid, ygrid[0], ygrid[-1], color="k", linewidth=hlinewidth)
    ax.vlines(ygrid, xgrid[0], xgrid[-1], color="k", linewidth=vlinewidth)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(which="both", bottom=False, left=False)
    if remove_ticklabels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if xlabel is not None:
        ax.set_xlabel(f"{xlabel}")
        if xlabel_pos is not None:
            ax.xaxis.set_label_coords(xlabel_pos[0], xlabel_pos[1])
    if ylabel is not None:
        ax.set_ylabel(f"{ylabel}")
        if ylabel_pos is not None:
            ax.yaxis.set_label_coords(ylabel_pos[0], ylabel_pos[1])
    return ax
