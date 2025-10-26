#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.m1_specialization import M1_STRATEGY_CONFIG
import pandas as pd
import numpy as np

from tools.m1_performance import M1PerformanceMonitor

def synth_loader(symbol, timeframe, count):
    n = max(200, count)
    idx = pd.date_range("2024-02-01", periods=n, freq="min")
    base = 1.2000
    noise = np.random.normal(0, 0.00004, size=n).cumsum()
    close = base + 0.000015 * np.arange(n) + noise
    df = pd.DataFrame({
        "open": close, "high": close + 0.00008, "low": close - 0.00008, "close": close, "spread": 0.00008
    }, index=idx)
    return df

def test_run_and_export_report(monkeypatch, tmp_path):
    # parchear loader del monitor
    import tools.m1_performance as mp
    monkeypatch.setattr(mp, "load_historical_data", synth_loader)

    monitor = M1PerformanceMonitor()
    report = monitor.run_specialized_backtest("TEST", "M1", days=5)
    assert "win_rate" in report and "profit_factor" in report and "total_trades" in report

    path = tmp_path / "session.json"
    # exporta lista con un reporte
    out_file = monitor.export_session_report([report], "M1", str(path))
    assert out_file.endswith(".json")