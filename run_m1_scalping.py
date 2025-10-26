#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner del sistema especializado en 1M con fallback a 5M si conviene.
"""

import logging
from tools.m1_performance import M1PerformanceMonitor
from config.m1_specialization import M1_SYMBOLS, TIMEFRAME_DEFAULT, PERFORMANCE_MONITOR_CONFIG

def main():
    logging.basicConfig(level=logging.INFO)
    monitor = M1PerformanceMonitor()

    # Selección de timeframe por warm-up usando el primer símbolo
    first_symbol = M1_SYMBOLS[0]
    best_tf, quick = monitor.quick_compare_timeframes(first_symbol, days=5)
    logging.info(f"WARM-UP {first_symbol}: {quick}")
    chosen_tf = best_tf if quick.get(best_tf, {}).get("win_rate", 0.0) >= PERFORMANCE_MONITOR_CONFIG["switch_timeframe_if_winrate_below"] else TIMEFRAME_DEFAULT
    logging.info(f"⏱️ Timeframe elegido para la sesión: {chosen_tf}")

    # Ejecutar sobre todos los mercados contemplados
    session_reports = []
    for sym in M1_SYMBOLS:
        report = monitor.run_specialized_backtest(sym, chosen_tf, days=15)
        session_reports.append(report)
        logging.info(f"{sym} {chosen_tf} → WinRate={report.get('win_rate', 0.0):.2%}, PF={report.get('profit_factor', 0.0):.2f}, Trades={report.get('total_trades', 0)}")

    # Resumen global
    valid = [r for r in session_reports if 'error' not in r]
    if not valid:
        logging.error("❌ No hay reportes válidos.")
        return
    avg_wr = sum(r['win_rate'] for r in valid) / len(valid)
    avg_pf = sum(r['profit_factor'] for r in valid) / len(valid)
    logging.info(f"📊 Sesión {chosen_tf} — Promedio WinRate={avg_wr:.2%}, ProfitFactor={avg_pf:.2f}")
    # Exportar reporte detallado de la sesión
    monitor.export_session_report(session_reports, chosen_tf)

if __name__ == "__main__":
    main()