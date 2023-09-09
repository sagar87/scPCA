import numpy as np
import pandas as pd
import pytest

from scpca import TimeSerie


@pytest.fixture
def ts_monthly_constant():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y=np.linspace(1, 1, len(index)))


@pytest.fixture
def ts_monthly_1():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y=np.random.randn(len(index)))


@pytest.fixture
def ts_monthly_2():
    start = "2020"
    end = "2021"
    freq = "1M"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y=np.random.randn(len(index)))


@pytest.fixture
def ts_daily():
    start = "2020"
    end = "2021"
    freq = "1D"
    index = pd.date_range(start=start, end=end, freq=freq)
    return TimeSerie(index=index, y=np.random.randn(len(index)))


def test_len(ts_monthly_1, ts_daily):
    assert len(ts_monthly_1) == 12
    # 2020 is a leap year and 2021-01-01 is included
    assert len(ts_daily) == 367


# def test_addition(ts_monthly_1, ts_monthly_2, ts_monthly_constant):
#     assert (ts_monthly_1 + ts_monthly_2) == (ts_monthly_2 + ts_monthly_1)
#     assert (ts_monthly_1 + ts_monthly_constant) == (ts_monthly_1 + 1)
#     # this line fails if we don't implement a __radd__ method:
#     assert (1 + ts_monthly_1) == (ts_monthly_1 + 1)
