#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motor de ejecución para 1M con aplicación de spread, slippage y costes.
Funciona en modo backtest; si se desea, puede integrarse con MT5Connector para live.
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

try:
    from mt5_connector.connector import MT5Connector  # opcional para live
except Exception:
    MT5Connector = None

class ExecutionEngineM1:
    def __init__(self, risk_manager, commission_pct: float = 0.00010):
        self.logger = logging.getLogger('ExecutionEngineM1')
        self.rm = risk_manager
        self.commission_pct = commission_pct

    def apply_costs(self, position: str, entry_price: float, exit_price: float,
                    lots: float, spread_price: float, slippage_pips: float) -> float:
        """
        Calcula P&L neto incluyendo spread, slippage y comisión.
        """
        pip_value = 10.0  # USD por pip por lote
        pips_move = (exit_price - entry_price) * 100000 if position == 'BUY' else (entry_price - exit_price) * 100000
        gross = pips_move * pip_value * lots

        spread_pips = spread_price * 100000
        commission = (lots * pip_value * self.commission_pct * abs(pips_move))  # aproximación proporcional
        total_cost = (spread_pips + slippage_pips) * pip_value * lots + commission

        return gross - total_cost

    def simulate_fill_price(self, position: str, price: float, slippage_pips: float) -> float:
        """
        Ajusta precio de ejecución por slippage: BUY empeora, SELL también.
        """
        adjust = (slippage_pips / 100000.0)
        return price + adjust if position == 'BUY' else price - adjust

    def backtest_loop(self, df: pd.DataFrame, strategy, symbol: str, config: Dict) -> Dict:
        trades: List[Dict] = []
        equity_curve: List[float] = [self.rm.state.equity]
        position = None
        entry_price = None
        entry_time = None
        entry_conf = 0.0
        entry_index = None
        max_bars = int(config.get('max_holding_bars', 15))

        for i in range(50, len(df)):
            # Evitar copias profundas: usar vista del DataFrame
            window = df.iloc[:i+1]  # vista, sin .copy()
            now = df.index[i]
            price = float(df['close'].iloc[i])
            spread = float(df['spread'].iloc[i]) if 'spread' in df.columns else 0.0

            signal, confidence = strategy.generate_signal(window)

            if position is None and signal in ['BUY', 'SELL'] and confidence >= float(config.get('min_confidence', 0.70)):
                if self.rm.should_stop_trading():
                    break
                sl, tp = strategy.calculate_dynamic_exits(window, signal, price)
                atr_val = float(df['atr'].iloc[i])
                lots = self.rm.compute_position_size(atr_val, price, sl)
                slippage = self.rm.slippage_pips(atr_val, confidence)
                fill_price = self.simulate_fill_price(signal, price, slippage)

                position = signal
                entry_price = fill_price
                entry_conf = confidence
                entry_time = now
                entry_index = i
                entry_sl = sl
                entry_tp = tp
                entry_spread = spread
                entry_lots = lots

            elif position is not None:
                atr_val = float(df['atr'].iloc[i])
                slippage = self.rm.slippage_pips(atr_val, entry_conf)
                exit_price = self.simulate_fill_price(position, price, slippage)
                force_exit = (i - entry_index) >= max_bars

                if position == 'BUY':
                    hit_sl = price <= entry_sl
                    hit_tp = price >= entry_tp
                else:
                    hit_sl = price >= entry_sl
                    hit_tp = price <= entry_tp

                if hit_sl or hit_tp or force_exit:
                    pnl = self.apply_costs(position, entry_price, exit_price, entry_lots, entry_spread, slippage)
                    self.rm.update_after_trade(pnl)
                    trades.append({
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": now,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "lots": entry_lots,
                        "profit": pnl,
                        "type": position,
                        "signal_confidence": entry_conf,
                        "entry_spread": entry_spread
                    })
                    equity_curve.append(self.rm.state.equity)
                    position = None
                    entry_index = None

        return {"trades": trades, "equity_curve": equity_curve, "final_equity": equity_curve[-1]}