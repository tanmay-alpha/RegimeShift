"""Consumer-visible chart values must retain first-day losses and entry fees."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_shift.plots import _drawdowns_chart, _equity_curves_chart


def test_charts_retain_initial_capital_loss(tmp_path, monkeypatch):
    returns = pd.Series([-0.20, 0.10], index=pd.bdate_range("2020-01-01", periods=2))
    result = SimpleNamespace(net_returns=returns, net_equity=(1 + returns).cumprod(),
                             gross_equity=(1 + returns).cumprod())
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda *args: None)
    try:
        _drawdowns_chart(result, {}, str(tmp_path))
        np.testing.assert_allclose(plt.gcf().axes[0].lines[0].get_ydata(), [-20.0, -12.0])
        close("all")
        _equity_curves_chart(result, {}, str(tmp_path))
        np.testing.assert_allclose(plt.gcf().axes[0].lines[0].get_ydata(), [0.8, 0.88])
    finally:
        close("all")
