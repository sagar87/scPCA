import math

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from scpca.time_serie import TimeSerie


def affine(start, end, freq, start_y, end_y):
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y=np.linspace(start_y, end_y, len(index)))


def constant(start, end, freq, value):
    return affine(start, end, freq, value, value)


def cosine(start, end, freq, amp=1, n_periods=1):
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y=amp * np.cos(np.linspace(0, 2 * math.pi * n_periods, num=len(index))),
    )


def sine(start, end, freq, n_periods=1):
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(
        index=index,
        y=np.sin(np.linspace(0, 2 * math.pi * n_periods, num=len(index))),
    )
