#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from tools.param_tuning_m1 import ParameterTunerM1
from config.m1_specialization import M1_STRATEGY_CONFIG

def synth_loader(symbol, timeframe, count):
    n = max(300, count)
    idx = pd.date_range("2024-04-01", periods=n, freq="min")
    base = 1.1500
    noise = np.random.normal(0, 0.00005, size=n).cumsum()
    close = base + 0.00002 * np.arange(n) + noise
    df = pd.DataFrame({
        "open": close, "high": close + 0.00010, "low": close - 0.00010, "close": close, "spread": 0.00010
    }, index=idx)
    return df

def test_grid_and_iterative_adjust():
    tuner = ParameterTunerM1(data_provider=synth_loader)
    grid = {
        "ema_fast": [3, 4],
        "ema_medium": [8, 10],
        "ema_slow": [18, 21],
        "min_confidence": [0.65, 0.70],
        "take_profit_multiplier": [1.6, 1.8],
        "stop_loss_multiplier": [0.9, 1.0]
    }
    res = tuner.grid_search("TEST", "M1", M1_STRATEGY_CONFIG.copy(), grid, days=5, max_combinations=8)
    assert "best" in res and res["tested"] > 0

    final_cfg, hist = tuner.iterative_adjust("TEST", "M1", M1_STRATEGY_CONFIG.copy(), days=5, iterations=2)
    assert isinstance(final_cfg, dict) and len(hist) >= 1