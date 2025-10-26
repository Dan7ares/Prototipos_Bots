#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.m1_performance import M1PerformanceMonitor
from config.m1_specialization import M1_SYMBOLS
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("M1Optimizer")

def max_drawdown(equity: List[float]) -> Tuple[float, List[float]]:
    if not equity:
        return 0.0, []
    peaks = []
    dds = []
    peak = equity[0]
    for val in equity:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0.0
        peaks.append(peak)
        dds.append(dd)
    return max(dds) if dds else 0.0, dds

def trade_analysis(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "total": 0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0,
            "avg_duration_minutes": 0.0,
            "consecutive_win_streak_max": 0,
            "consecutive_loss_streak_max": 0,
            "hourly_distribution": {}
        }
    profits = [t["profit"] for t in trades]
    best = max(profits)
    worst = min(profits)
    durations = []
    win_streak = loss_streak = 0
    max_win_streak = max_loss_streak = 0
    hourly = {}

    for t in trades:
        try:
            et = pd.to_datetime(t["entry_time"])
            xt = pd.to_datetime(t["exit_time"])
            durations.append((xt - et).total_seconds() / 60.0)
            hour = int(et.hour)
            hourly[hour] = hourly.get(hour, 0) + 1
        except Exception:
            pass
        if t["profit"] > 0:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)

    return {
        "total": len(trades),
        "best_trade_usd": float(best),
        "worst_trade_usd": float(worst),
        "avg_duration_minutes": float(np.mean(durations) if durations else 0.0),
        "consecutive_win_streak_max": int(max_win_streak),
        "consecutive_loss_streak_max": int(max_loss_streak),
        "hourly_distribution": hourly
    }

def rr_ratio(avg_win: float, avg_loss: float) -> float:
    if avg_loss >= 0:
        return 0.0
    return abs(avg_win / abs(avg_loss)) if avg_win > 0 else 0.0

def evaluate(results: List[Dict], dd_threshold: float = 0.12, win_rate_target: float = 0.70) -> Dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"pass": False, "reason": "Sin resultados válidos", "summary": {}}

    # Consolidar métricas ponderadas por número de trades
    total_trades = sum(r["total_trades"] for r in valid)
    weighted_win = sum(r["win_rate"] * r["total_trades"] for r in valid) / total_trades if total_trades else 0.0
    weighted_pf = sum(r["profit_factor"] * r["total_trades"] for r in valid) / total_trades if total_trades else 0.0

    # Equity combinada (concat normalizada por capital inicial)
    combined_equity = []
    for r in valid:
        eq = r.get("equity_curve", [])
        if eq:
            # Normalización: arranque en el primer valor
            base = eq[0] if eq[0] else 1.0
            combined_equity.extend([v / base for v in eq])

    dd, _ = max_drawdown(combined_equity)
    avg_win = np.mean([r["avg_win"] for r in valid if r["avg_win"] > 0]) if valid else 0.0
    avg_loss = np.mean([r["avg_loss"] for r in valid if r["avg_loss"] < 0]) if valid else 0.0
    rr = rr_ratio(avg_win, avg_loss)

    criteria = {
        "win_rate_ok": weighted_win >= win_rate_target,
        "drawdown_ok": dd <= dd_threshold,
        "rr_ok": rr >= 1.5
    }

    all_ok = all(criteria.values())
    reason = "OK" if all_ok else f"Falla criterios: {', '.join([k for k,v in criteria.items() if not v])}"
    return {
        "pass": all_ok,
        "reason": reason,
        "summary": {
            "weighted_win_rate": weighted_win,
            "weighted_profit_factor": weighted_pf,
            "max_drawdown": dd,
            "risk_reward_ratio": rr,
            "total_trades": total_trades
        },
        "criteria": criteria
    }

def save_outputs(results: List[Dict], evaluation: Dict, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"m1_optimized_report_{timestamp}.json")
    csv_path = os.path.join(out_dir, f"m1_optimized_trades_{timestamp}.csv")
    eq_path = os.path.join(out_dir, f"m1_optimized_equity_{timestamp}.png")

    # JSON
    payload = {
        "timestamp": timestamp,
        "results": results,
        "evaluation": evaluation
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.isoformat() if hasattr(o, 'isoformat') else str(o))

    # CSV trades (aplanado)
    rows = []
    for r in results:
        if "error" in r:
            continue
        for t in r.get("trades", []):
            rows.append({
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "entry_time": t.get("entry_time"),
                "exit_time": t.get("exit_time"),
                "type": t.get("type"),
                "lots": t.get("lots"),
                "profit": t.get("profit"),
                "entry_price": t.get("entry_price"),
                "exit_price": t.get("exit_price"),
                "signal_confidence": t.get("signal_confidence"),
                "entry_spread": t.get("entry_spread")
            })
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

    # Equity plot: símbolo con mayor win rate
    best = None
    for r in results:
        if "error" in r:
            continue
        if best is None or r["win_rate"] > best["win_rate"]:
            best = r
    if best and best.get("equity_curve"):
        plt.figure(figsize=(10, 4))
        plt.plot(best["equity_curve"], label=f"{best['symbol']} {best['timeframe']}")
        plt.title("Equity Curve (mejor símbolo por win rate)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eq_path)

    return {"json": json_path, "csv": csv_path, "equity_png": eq_path}

def main():
    ap = argparse.ArgumentParser(description="Optimización M1 con datos reales y reporte ejecutivo")
    ap.add_argument("--days", type=int, default=15, help="Días de datos reales a cargar")
    ap.add_argument("--symbols", type=str, default=",".join(M1_SYMBOLS), help="Símbolos separados por coma")
    ap.add_argument("--output", type=str, default=".", help="Directorio de salida")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    monitor = M1PerformanceMonitor()

    results = []
    for sym in symbols:
        logger.info(f"▶ Ejecutando backtest M1 para {sym} con {args.days} días reales...")
        out = monitor.run_specialized_backtest(sym, "M1", days=args.days)
        if "error" in out:
            logger.warning(f"{sym}: {out['error']}")
        results.append(out)

    # Evaluación de criterios
    evaluation = evaluate(results, dd_threshold=0.12, win_rate_target=0.70)
    logger.info(f"Resumen: {evaluation['summary']}")
    logger.info(f"Criterios: {evaluation['criteria']}")
    logger.info(f"Resultado: {'✅ Cumple' if evaluation['pass'] else '❌ No cumple'} ({evaluation['reason']})")

    # Análisis de operaciones agregado
    all_trades = []
    for r in results:
        if "error" not in r:
            all_trades.extend(r.get("trades", []))
    trade_stats = trade_analysis(all_trades)
    logger.info(f"Trades: {trade_stats}")

    # Guardar reportes
    paths = save_outputs(results, {**evaluation, "trade_analysis": trade_stats}, args.output)
    logger.info(f"Reportes guardados: {paths}")

    # Recomendaciones básicas
    if not evaluation["pass"]:
        logger.info("Recomendaciones de ajuste:")
        if not evaluation["criteria"]["win_rate_ok"]:
            logger.info("- Subir 'min_confidence' o reforzar filtros ADX/Bollinger.")
        if not evaluation["criteria"]["drawdown_ok"]:
            logger.info("- Reducir 'risk_per_trade' o acortar 'max_holding_bars'.")
        if not evaluation["criteria"]["rr_ok"]:
            logger.info("- Incrementar 'take_profit_multiplier' o ajustar 'stop_loss_multiplier'.")

if __name__ == "__main__":
    main()