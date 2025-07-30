import pandas as pd

from thetagang.stock_strategies import compute_bx_trender, bx_trender_signal


def test_bx_trender_signal_buy() -> None:
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    osc = compute_bx_trender(data)
    # create a series crossing above zero by appending a low value then high
    osc.iloc[-2] = -1
    osc.iloc[-1] = 1
    assert bx_trender_signal(osc) == "BUY"


def test_bx_trender_signal_sell() -> None:
    data = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    osc = compute_bx_trender(data)
    osc.iloc[-2] = 1
    osc.iloc[-1] = -1
    assert bx_trender_signal(osc) == "SELL"
