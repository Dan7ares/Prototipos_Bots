#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
# Añade el raíz del proyecto al sys.path (importa core/strategies/tools)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import time

from strategies.m1_strategy import M1ScalpingStrategy
from core.execution_m1 import ExecutionEngineM1
from core.risk_manager_m1 import RiskManagerM1
from config.m1_specialization import M1_STRATEGY_CONFIG, RISK_CONFIG_M1

def make_synth(n=300, trend=0.00002, spread=0.00010):
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    base = 1.1000
    noise = np.random.normal(0, 0.00005, size=n).cumsum()
    close = base + trend * np.arange(n) + noise
    high = close + 0.00010
    low = close - 0.00010
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "spread": spread}, index=idx)
    return df

def test_backtest_loop_perf_and_metrics():
    df = make_synth()
    strat = M1ScalpingStrategy(M1_STRATEGY_CONFIG)
    df = strat.calculate_indicators(df)
    rm = RiskManagerM1(RISK_CONFIG_M1)
    engine = ExecutionEngineM1(rm)

    t0 = time.time()
    out = engine.backtest_loop(df, strat, "TEST", M1_STRATEGY_CONFIG)
    elapsed = time.time() - t0

    assert "trades" in out and "equity_curve" in out and "final_equity" in out
    assert elapsed < 5.0  # valida que no hay bloqueo por copias
    assert isinstance(out["final_equity"], (float, int))