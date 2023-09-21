from collections import OrderedDict
from typing import Dict, NamedTuple

import numpy as np
from numpy.typing import NDArray
from patsy.design_info import DesignMatrix  # type: ignore


class StateMapping(NamedTuple):
    mapping: Dict[str, int]
    reverse: Dict[int, str]
    encoding: NDArray[np.float32]
    idx: NDArray[np.int64]
    columns: Dict[int, str]
    states: Dict[str, int]
    sparse: Dict[str, int]


def _get_states(design: DesignMatrix) -> StateMapping:
    """Extracts the states from the design matrix.

    Parameters
    ----------
    design: DesignMatrix
        Design matrix of the model.

    Returns
    -------
    StateMapping: namedtuple
        Named tuple with the following fields
    """
    unique_rows, inverse_rows = np.unique(np.asarray(design), axis=0, return_inverse=True)

    combinations = OrderedDict()
    sparse_state = {}
    for j, row in enumerate(range(unique_rows.shape[0])):
        idx = tuple(np.where(unique_rows[row] == 1)[0])
        combinations[idx] = unique_rows[row], j

        state_name = "|".join([design.design_info.column_names[i] for i in np.where(unique_rows[row] == 1)[0]])
        if state_name != "Intercept":
            state_name = state_name.lstrip("Intercept|")
        sparse_state[state_name] = j

    factor_cols = {v: k for k, v, in design.design_info.column_name_indexes.items()}
    state_cols = {v: k for k, v in factor_cols.items()}

    state_mapping = {}
    reverse_mapping = {}
    for idx, (key, value) in enumerate(combinations.items()):  # type: ignore
        state = ""
        for idx in key:
            state += factor_cols[idx] + "|"
        state = state.rstrip("|")
        state_mapping[state] = value[1]
        reverse_mapping[value[1]] = state

    return StateMapping(
        state_mapping, reverse_mapping, unique_rows, inverse_rows, factor_cols, state_cols, sparse_state
    )
