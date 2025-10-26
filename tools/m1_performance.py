#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitor y backtest especializado en 1M con selección automática M1/M5 según win rate inicial.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from core.data_loader import load_historical_data
from strategies.m1_strategy import M1ScalpingStrategy
from core.risk_manager_m1 import RiskManagerM1
from core.execution_m1 import ExecutionEngineM1
from config.m1_specialization import (
    M1_STRATEGY_CONFIG, M5_STRATEGY_CONFIG,
    RISK_CONFIG_M1, M1_SYMBOLS,
    PERFORMANCE_MONITOR_CONFIG, TIMEFRAME_DEFAULT
)

class M1PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('M1PerformanceMonitor')

    def _json_default(self, o):
        # Normaliza objetos no-serializables para JSON
        import numpy as np
        import pandas as pd
        from datetime import datetime
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        # Arrays y similares
        if hasattr(o, "tolist"):
            try:
                return o.tolist()
            except Exception:
                pass
        # Fallback seguro
        return str(o)

    def quick_compare_timeframes(self, symbol: str, days: int = 5) -> Tuple[str, Dict]:
        """
        Ejecuta un warm-up rápido en M1 y M5 y elige el mejor según win rate.
        """
        results = {}
        for tf, cfg in [("M1", M1_STRATEGY_CONFIG), ("M5", M5_STRATEGY_CONFIG)]:
            df = load_historical_data(symbol, tf, count=days * (1440 // (1 if tf == "M1" else 5)))
            if df is None or len(df) < 120:
                results[tf] = {"win_rate": 0.0, "total_trades": 0}
                continue
            strat = M1ScalpingStrategy(cfg)
            df = strat.calculate_indicators(df)
            rm = RiskManagerM1(RISK_CONFIG_M1)
            engine = ExecutionEngineM1(rm, commission_pct=RISK_CONFIG_M1.get("commission_pct", 0.00010))
            out = engine.backtest_loop(df, strat, symbol, cfg)
            wins = sum(1 for t in out["trades"] if t["profit"] > 0)
            total = len(out["trades"])
            wr = (wins / total) if total else 0.0
            results[tf] = {"win_rate": wr, "total_trades": total, "final_equity": out["final_equity"]}
        best_tf = max(results.keys(), key=lambda k: results[k]["win_rate"])
        return best_tf, results

    def run_specialized_backtest(self, symbol: str, timeframe: str, days: int = 15) -> Dict:
        cfg = M1_STRATEGY_CONFIG if timeframe == "M1" else M5_STRATEGY_CONFIG
        df = load_historical_data(symbol, timeframe, count=days * (1440 // (1 if timeframe == "M1" else 5)))
        if df is None or len(df) < 120:
            return {"error": f"Datos insuficientes para {symbol} {timeframe}"}
        strat = M1ScalpingStrategy(cfg)
        df = strat.calculate_indicators(df)
        rm = RiskManagerM1(RISK_CONFIG_M1)
        engine = ExecutionEngineM1(rm, commission_pct=RISK_CONFIG_M1.get("commission_pct", 0.00010))
        out = engine.backtest_loop(df, strat, symbol, cfg)

        # Métricas clave
        trades = out["trades"]
        equity = out["equity_curve"]
        wins = sum(1 for t in trades if t["profit"] > 0)
        losses = sum(1 for t in trades if t["profit"] <= 0)
        total = len(trades)
        win_rate = (wins / total) if total else 0.0
        gross_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
        gross_loss = abs(sum(t["profit"] for t in trades if t["profit"] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
        # Sharpe aprox por trade
        returns = np.diff(equity) / np.clip(equity[:-1], 1e-9, None) if len(equity) > 1 else np.array([])
        sharpe = float((np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(len(returns))) if returns.size > 1 and np.std(returns, ddof=1) > 0 else 0.0
        avg_win = np.mean([t["profit"] for t in trades if t["profit"] > 0]) if wins else 0.0
        avg_loss = np.mean([t["profit"] for t in trades if t["profit"] <= 0]) if losses else 0.0
        expectancy = float(win_rate * avg_win + (1.0 - win_rate) * (-avg_loss))

        # Visualización rápida
        try:
            plt.figure(figsize=(9, 4))
            plt.plot(equity, label=f'Equity ({symbol} {timeframe})')
            plt.title(f'Equity Curve {symbol} {timeframe}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{symbol}_{timeframe}_equity.png')
        except Exception:
            pass

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_trades": total,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "final_equity": out["final_equity"],
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe,
            "expectancy": expectancy,
            "equity_curve": equity,
            "trades": trades
        }

    def export_session_report(self, session_reports, timeframe: str, out_path: str | None = None) -> str:
        import json
        from datetime import datetime
        valid = [r for r in session_reports if 'error' not in r]
        summary = {
            "timeframe": timeframe,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "symbols": [r["symbol"] for r in valid],
            "avg_win_rate": sum(r["win_rate"] for r in valid) / len(valid) if valid else 0.0,
            "avg_profit_factor": sum(r["profit_factor"] for r in valid) / len(valid) if valid else 0.0,
            "reports": valid
        }
        fname = out_path or f"performance_comparison_{summary['timestamp']}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=self._json_default)
        self.logger.info(f"💾 Reporte exportado a {fname}")
        return fname